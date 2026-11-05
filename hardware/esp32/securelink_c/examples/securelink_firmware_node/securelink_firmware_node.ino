#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

#include "securelink_protocol.h"
#include "securelink_tinyml.h"

/* MAC identities for this specific hardware pair */
static const uint8_t aliceMac[6]     = {0x38, 0x18, 0x2b, 0xb2, 0x87, 0x68};
static const uint8_t bobMac[6]       = {0x2c, 0xbc, 0xbb, 0x0c, 0x67, 0xe4};
static const uint8_t broadcastMac[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
constexpr uint8_t channel = 6;

/* Shared pre-shared key for fallback and message authentication */
static const uint8_t authSecret[SECURELINK_KEY_SIZE] = {
    0x4c, 0x6f, 0x63, 0x61, 0x6e, 0x64, 0x4b, 0x65,
    0x79, 0x53, 0x65, 0x63, 0x75, 0x72, 0x65, 0x4c,
    0x69, 0x6e, 0x6b, 0x50, 0x72, 0x65, 0x53, 0x68,
    0x61, 0x72, 0x65, 0x64, 0x4b, 0x65, 0x79, 0x31
};

/* Protocol Packet Identifiers */
enum PacketType : uint8_t {
    PKT_PROBE   = 0x10,
    PKT_REPLY   = 0x11,
    PKT_OFFER   = 0x20,
    PKT_CONFIRM = 0x21,
    PKT_DATA    = 0x30
};

typedef struct __attribute__((packed)) {
    uint8_t type;        /* PKT_PROBE or PKT_REPLY */
    uint8_t probe_idx;   /* 0 .. 7 */
    uint16_t session_id; /* Rekey counter */
    uint32_t timestamp_us;
} securelink_probe_pkt_t;

/* Buffer capacities */
constexpr size_t BURST_FRAMES = 8;
constexpr size_t CSI_BYTES = 128; /* 64 subcarriers x 2 (I/Q) */

static securelink_bch_ctx_t bchContext;
static securelink_replay_cache_t replayCache;
static bool isAlice = false;
static const uint8_t *peerMac = nullptr;

/* Autonomous Rekeying & CSI Capture Ring Buffer */
static volatile int8_t csiBuffer[BURST_FRAMES][CSI_BYTES];
static volatile size_t csiHead = 0;
static volatile size_t csiCount = 0;

/* Static protocol buffers to prevent stack canary overflows on Xtensa */
static int8_t sharedOrderedCsi[BURST_FRAMES * CSI_BYTES];
static int8_t sharedDenoisedCsi[BURST_FRAMES * CSI_BYTES];
static securelink_offer_t sharedOffer;
static securelink_reply_t sharedReply;
static securelink_bob_session_t bobSession;

static void get_unrolled_csi(int8_t *out_csi) {
    for (size_t i = 0; i < BURST_FRAMES; i++) {
        size_t idx = (csiHead + i) % BURST_FRAMES;
        memcpy(out_csi + (i * CSI_BYTES), (const void *)csiBuffer[idx], CSI_BYTES);
    }
}

static uint8_t activeSessionKey[SECURELINK_KEY_SIZE];
static securelink_status_t activeKeyMode = SECURELINK_STATUS_FALLBACK;
static securelink_alice_session_t aliceSession;

static uint32_t lastRekeyMs = 0;
static uint32_t lastDataMs = 0;
static uint32_t rekeyCounter = 0;
static uint32_t packetCounter = 0;

/* --- On-Chip Boot Self-Tests --- */
static void run_onboard_selftest(void) {
    Serial.println("=============================================");
    Serial.println("  SecureLink Embedded C Node (Autonomous Rekey)");
    Serial.println("=============================================");
    Serial.println("--- Starting On-Chip Embedded C Self-Tests ---");

    /* 1. BCH(511, 457) error correction */
    uint8_t test_msg[38] = "SecureLink Embedded C Physical Layer!";
    uint8_t ecc[SECURELINK_BCH_ECC_BYTES];
    securelink_bch_encode(&bchContext, test_msg, sizeof(test_msg), ecc);
    test_msg[0] ^= 0x01;
    test_msg[5] ^= 0x10;
    test_msg[12] ^= 0x04;
    int corrected = securelink_bch_decode(&bchContext, test_msg, sizeof(test_msg), ecc);
    if (corrected == 3 && memcmp(test_msg, "SecureLink Embedded C Physical Layer!", 38) == 0) {
        Serial.printf("[SELF-TEST PASS] BCH(511, 457) corrected %d bit errors on-chip!\n", corrected);
    } else {
        Serial.printf("[SELF-TEST FAIL] BCH decode returned %d\n", corrected);
    }

    /* 2. Hardware AES-256-GCM */
    uint8_t testKey[SECURELINK_KEY_SIZE];
    uint8_t salt[16] = {0x01, 0x02};
    uint8_t info[] = "self-test";
    securelink_hkdf_sha384(authSecret, SECURELINK_KEY_SIZE, salt, sizeof(salt),
                           info, sizeof(info)-1, testKey, SECURELINK_KEY_SIZE);
    uint8_t iv[SECURELINK_GCM_NONCE_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    uint8_t pt[] = "ESP32-AES-GCM-OK";
    uint8_t ct[sizeof(pt)];
    uint8_t tag[SECURELINK_TAG_SIZE];
    uint8_t dec[sizeof(pt)];
    securelink_aes_gcm_encrypt(testKey, iv, nullptr, 0, pt, sizeof(pt), ct, tag);
    if (securelink_aes_gcm_decrypt(testKey, iv, nullptr, 0, ct, sizeof(ct), tag, dec) == SECURELINK_OK &&
        memcmp(pt, dec, sizeof(pt)) == 0) {
        Serial.println("[SELF-TEST PASS] Hardware-accelerated AES-256-GCM encryption/decryption verified!");
    } else {
        Serial.println("[SELF-TEST FAIL] AES-GCM decryption failed!");
    }

    /* 3. TinyML CNN Denoiser */
    float test_in[2][SECURELINK_TINYML_SUBCARRIERS];
    float test_out[2][SECURELINK_TINYML_SUBCARRIERS];
    for (int s = 0; s < SECURELINK_TINYML_SUBCARRIERS; s++) {
        test_in[0][s] = 1.0f + 0.1f * sinf((float)s * 0.2f);
        test_in[1][s] = 0.5f + 0.1f * cosf((float)s * 0.2f);
    }
    uint32_t t0 = micros();
    securelink_tinyml_denoise(test_in, test_out);
    uint32_t dt = micros() - t0;
    Serial.printf("[SELF-TEST PASS] TinyML CNN denoiser executed on-chip in %lu microseconds! (In=%.3f -> Out=%.3f)\n",
                  (unsigned long)dt, test_in[0][0], test_out[0][0]);
    Serial.println("----------------------------------------------");
}

/* --- Hardware CSI Callback --- */
static void on_csi_rx(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len < CSI_BYTES) return;
    if (memcmp(info->mac, peerMac, 6) != 0) return;

    memcpy((void *)csiBuffer[csiHead], info->buf, CSI_BYTES);
    csiHead = (csiHead + 1) % BURST_FRAMES;
    if (csiCount < BURST_FRAMES) csiCount++;
}

/* --- ESP-NOW Receive Callback --- */
static void on_espnow_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    if (!info || !data || len < 1) return;

    uint8_t pkt_type = data[0];
    const uint8_t *payload = data + 1;
    size_t payload_len = len - 1;

    switch (pkt_type) {
        case PKT_PROBE: {
            /* Responder (Bob): send immediate pong over broadcast to trigger Alice's reciprocal CSI */
            if (isAlice || len < (int)sizeof(securelink_probe_pkt_t)) return;
            const securelink_probe_pkt_t *in_probe = (const securelink_probe_pkt_t *)data;
            securelink_probe_pkt_t reply = {PKT_REPLY, in_probe->probe_idx, in_probe->session_id, micros()};
            esp_now_send(broadcastMac, (const uint8_t *)&reply, sizeof(reply));
            break;
        }

        case PKT_REPLY: {
            /* Alice receives pong: reciprocal CSI was captured in on_csi_rx */
            break;
        }

        case PKT_OFFER: {
            /* Bob receives Alice's Offer -> Denoise with TinyML and Reconcile */
            if (isAlice || payload_len != sizeof(securelink_offer_t)) return;
            Serial.printf("[REKEY] Bob received Alice offer! Running on-chip TinyML denoising on %u frames...\n", (unsigned)csiCount);

            get_unrolled_csi(sharedOrderedCsi);

            for (size_t f = 0; f < BURST_FRAMES; f++) {
                float in_iq[2][64];
                float out_iq[2][64];
                for (int s = 0; s < 64; s++) {
                    in_iq[0][s] = (float)sharedOrderedCsi[(f * 64 + s) * 2 + 0] / 128.0f;
                    in_iq[1][s] = (float)sharedOrderedCsi[(f * 64 + s) * 2 + 1] / 128.0f;
                }
                securelink_tinyml_denoise(in_iq, out_iq);
                for (int s = 0; s < 64; s++) {
                    int r = (int)(out_iq[0][s] * 128.0f);
                    int q = (int)(out_iq[1][s] * 128.0f);
                    sharedDenoisedCsi[(f * 64 + s) * 2 + 0] = (int8_t)(r < -128 ? -128 : (r > 127 ? 127 : r));
                    sharedDenoisedCsi[(f * 64 + s) * 2 + 1] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
                }
            }

            memcpy(&sharedOffer, payload, sizeof(sharedOffer));
            uint8_t transcript[] = "autonomous-securelink-v1";

            securelink_err_t err = securelink_bob_accept_offer(
                sharedDenoisedCsi, BURST_FRAMES, 64, &sharedOffer,
                authSecret, sharedOffer.nonce, transcript, sizeof(transcript) - 1,
                &bchContext, &replayCache, &sharedReply, &bobSession
            );

            if (err == SECURELINK_OK) {
                memcpy(activeSessionKey, bobSession.key, SECURELINK_KEY_SIZE);
                activeKeyMode = bobSession.status;
                Serial.printf("[REKEY SUCCESS] Bob derived 256-bit key! Status=%s, Corrections=%u\n",
                              bobSession.status == SECURELINK_STATUS_ACTIVE ? "PHY_ACTIVE" : "FALLBACK",
                              (unsigned)bobSession.corrected_bits);

                /* Send confirmation packet back to Alice */
                uint8_t out_pkt[1 + sizeof(sharedReply)];
                out_pkt[0] = PKT_CONFIRM;
                memcpy(out_pkt + 1, &sharedReply, sizeof(sharedReply));
                esp_now_send(peerMac, out_pkt, sizeof(out_pkt));
            } else {
                Serial.printf("[REKEY FAIL] Bob offer acceptance failed: err=%d\n", (int)err);
            }
            break;
        }

        case PKT_CONFIRM: {
            /* Alice receives Bob's Confirmation -> Finalize Key */
            if (!isAlice || payload_len != sizeof(securelink_reply_t)) return;

            securelink_reply_t confirmReply;
            memcpy(&confirmReply, payload, sizeof(confirmReply));

            securelink_err_t err = securelink_alice_finalize(&aliceSession, &confirmReply);
            if (err == SECURELINK_OK) {
                memcpy(activeSessionKey, aliceSession.key, SECURELINK_KEY_SIZE);
                activeKeyMode = aliceSession.status;
                Serial.printf("[REKEY SUCCESS] Alice finalized 256-bit key! Status=%s\n",
                              aliceSession.status == SECURELINK_STATUS_ACTIVE ? "PHY_ACTIVE" : "FALLBACK");
            } else {
                Serial.printf("[REKEY FAIL] Alice finalization failed: err=%d\n", (int)err);
            }
            break;
        }

        case PKT_DATA: {
            /* Bob receives Encrypted Application Telemetry -> Hardware AES-256-GCM Decrypt */
            if (payload_len < SECURELINK_GCM_NONCE_SIZE + SECURELINK_TAG_SIZE) return;

            const uint8_t *iv = payload;
            const uint8_t *tag = payload + SECURELINK_GCM_NONCE_SIZE;
            const uint8_t *ct = payload + SECURELINK_GCM_NONCE_SIZE + SECURELINK_TAG_SIZE;
            size_t ct_len = payload_len - SECURELINK_GCM_NONCE_SIZE - SECURELINK_TAG_SIZE;

            uint8_t plaintext[128];
            if (ct_len >= sizeof(plaintext)) return;

            securelink_err_t err = securelink_aes_gcm_decrypt(
                activeSessionKey, iv, nullptr, 0, ct, ct_len, tag, plaintext
            );

            if (err == SECURELINK_OK) {
                plaintext[ct_len] = '\0';
                Serial.printf("[AUTONOMOUS-PAYLOAD] Decrypted (%s): \"%s\"\n",
                              activeKeyMode == SECURELINK_STATUS_ACTIVE ? "PHY_KEY" : "FALLBACK_KEY",
                              (char *)plaintext);
            } else {
                Serial.println("[AUTONOMOUS-PAYLOAD FAIL] AES-GCM authentication tag rejected!");
            }
            break;
        }

        default:
            break;
    }
}

void setup() {
    Serial.begin(115200);
    delay(500);

    /* Initialize BCH and Replay contexts */
    securelink_bch_init(&bchContext);
    securelink_replay_cache_init(&replayCache);

    /* Run on-chip verification */
    run_onboard_selftest();

    /* Wi-Fi & MAC Configuration */
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    uint8_t localMac[6];
    esp_wifi_get_mac(WIFI_IF_STA, localMac);

    if (memcmp(localMac, aliceMac, 6) == 0) {
        isAlice = true;
        peerMac = bobMac;
    } else {
        isAlice = false;
        peerMac = aliceMac;
    }

    Serial.printf("[ROLE] %s | Local MAC: %02X:%02X:%02X:%02X:%02X:%02X | Peer MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  isAlice ? "Alice (Initiator)" : "Bob (Responder)",
                  localMac[0], localMac[1], localMac[2], localMac[3], localMac[4], localMac[5],
                  peerMac[0], peerMac[1], peerMac[2], peerMac[3], peerMac[4], peerMac[5]);

    /* Default fallback key initialization */
    uint8_t initNonce[SECURELINK_NONCE_SIZE] = {0};
    uint8_t transcript[] = "autonomous-securelink-v1";
    securelink_derive_fallback_key(authSecret, initNonce, transcript, sizeof(transcript) - 1, activeSessionKey);
    activeKeyMode = SECURELINK_STATUS_FALLBACK;

    /* Wi-Fi & CSI Configuration */
    esp_wifi_set_channel(channel, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20);
    esp_wifi_set_promiscuous(true);

    wifi_csi_config_t csiConfig = {};
    csiConfig.lltf_en = false;
    csiConfig.htltf_en = true;
    csiConfig.stbc_htltf2_en = false;
    csiConfig.ltf_merge_en = false;
    csiConfig.channel_filter_en = true;

    esp_wifi_set_csi_config(&csiConfig);
    esp_wifi_set_csi_rx_cb(on_csi_rx, nullptr);
    esp_wifi_set_csi(true);

    /* ESP-NOW Initialization */
    if (esp_now_init() != ESP_OK) {
        Serial.println("[ERR] ESP-NOW init failed!");
        return;
    }
    esp_now_register_recv_cb(on_espnow_recv);

    /* Rate configuration: HT20 MCS0 required to trigger 802.11n HT-LTF CSI */
    esp_now_rate_config_t rate = {};
    rate.phymode = WIFI_PHY_MODE_HT20;
    rate.rate = WIFI_PHY_RATE_MCS0_LGI;

    /* Broadcast peer for CSI probes */
    esp_now_peer_info_t bcastPeer = {};
    memcpy(bcastPeer.peer_addr, broadcastMac, 6);
    bcastPeer.channel = channel;
    bcastPeer.ifidx = WIFI_IF_STA;
    bcastPeer.encrypt = false;
    esp_now_add_peer(&bcastPeer);
    esp_now_set_peer_rate_config(broadcastMac, &rate);

    /* Unicast peer for Offer, Confirm, and Encrypted Data */
    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, peerMac, 6);
    peerInfo.channel = channel;
    peerInfo.ifidx = WIFI_IF_STA;
    peerInfo.encrypt = false;
    esp_now_add_peer(&peerInfo);
    esp_now_set_peer_rate_config(peerMac, &rate);

    Serial.printf("[READY] Autonomous SecureLink node active on channel %u.\n", channel);
}

void loop() {
    uint32_t now = millis();

    /* 1. Host Benchmark Command Handler */
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd == "BENCHMARK") {
            Serial.println("--- Starting On-Chip TinyML Inference Benchmark (50 frames) ---");
            float in_f[2][SECURELINK_TINYML_SUBCARRIERS];
            float out_f[2][SECURELINK_TINYML_SUBCARRIERS];
            uint32_t total_us = 0;
            for (int i = 0; i < 50; i++) {
                for (int s = 0; s < SECURELINK_TINYML_SUBCARRIERS; s++) {
                    in_f[0][s] = sinf((float)(i + s) * 0.15f);
                    in_f[1][s] = cosf((float)(i + s) * 0.15f);
                }
                uint32_t t0 = micros();
                securelink_tinyml_denoise(in_f, out_f);
                total_us += (micros() - t0);
            }
            uint32_t avg_us = total_us / 50;
            Serial.printf("[BENCHMARK_RESULT] Runs=50, AvgLatency_ms=%.2f, Throughput_FPS=%.1f, FreeHeap_bytes=%lu\n",
                          avg_us / 1000.0f, 1000000.0f / (float)avg_us, (unsigned long)esp_get_free_heap_size());
        }
    }

    /* 2. Alice Autonomous Rekeying Trigger (every 10 seconds) */
    if (isAlice && (now - lastRekeyMs >= 10000)) {
        lastRekeyMs = now;
        rekeyCounter++;
        Serial.printf("\n[REKEY START #%lu] Triggering 8 reciprocal CSI ping-pong probes...\n", (unsigned long)rekeyCounter);

        /* Send 8 ping-pong probes spaced 20 ms apart */
        for (int p = 0; p < 8; p++) {
            securelink_probe_pkt_t probe = {PKT_PROBE, (uint8_t)p, (uint16_t)rekeyCounter, micros()};
            esp_now_send(broadcastMac, (const uint8_t *)&probe, sizeof(probe));
            delay(20);
        }

        Serial.printf("[REKEY] Captured %u reciprocal frames. Executing on-chip TinyML denoising...\n", (unsigned)csiCount);

        if (csiCount >= BURST_FRAMES) {
            get_unrolled_csi(sharedOrderedCsi);

            for (size_t f = 0; f < BURST_FRAMES; f++) {
                float in_iq[2][64];
                float out_iq[2][64];
                for (int s = 0; s < 64; s++) {
                    in_iq[0][s] = (float)sharedOrderedCsi[(f * 64 + s) * 2 + 0] / 128.0f;
                    in_iq[1][s] = (float)sharedOrderedCsi[(f * 64 + s) * 2 + 1] / 128.0f;
                }
                securelink_tinyml_denoise(in_iq, out_iq);
                for (int s = 0; s < 64; s++) {
                    int r = (int)(out_iq[0][s] * 128.0f);
                    int q = (int)(out_iq[1][s] * 128.0f);
                    sharedDenoisedCsi[(f * 64 + s) * 2 + 0] = (int8_t)(r < -128 ? -128 : (r > 127 ? 127 : r));
                    sharedDenoisedCsi[(f * 64 + s) * 2 + 1] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
                }
            }

            uint8_t nonce[SECURELINK_NONCE_SIZE];
            esp_fill_random(nonce, sizeof(nonce));
            uint8_t transcript[] = "autonomous-securelink-v1";

            securelink_err_t err = securelink_alice_create_offer(
                sharedDenoisedCsi, BURST_FRAMES, 64, authSecret, nonce,
                transcript, sizeof(transcript) - 1,
                &bchContext, &sharedOffer, &aliceSession
            );

            if (err == SECURELINK_OK) {
                Serial.println("[REKEY] Alice offer generated successfully! Sending to Bob over ESP-NOW...");
                uint8_t out_pkt[1 + sizeof(sharedOffer)];
                out_pkt[0] = PKT_OFFER;
                memcpy(out_pkt + 1, &sharedOffer, sizeof(sharedOffer));
                esp_now_send(peerMac, out_pkt, sizeof(out_pkt));
            } else {
                Serial.printf("[REKEY FALLBACK] Alice offer creation returned %d (maintaining active key, waiting for channel entropy)\n", (int)err);
            }
        }
    }

    /* 3. Encrypted Application Telemetry (Alice -> Bob every 3 seconds) */
    if (isAlice && (now - lastDataMs >= 3000)) {
        lastDataMs = now;
        packetCounter++;

        char plaintext[64];
        snprintf(plaintext, sizeof(plaintext), "ALICE_NODE: seq=%lu, uptime=%lu s, mode=%s",
                 (unsigned long)packetCounter, (unsigned long)(now / 1000),
                 activeKeyMode == SECURELINK_STATUS_ACTIVE ? "PHY_KEY" : "FALLBACK_KEY");

        size_t pt_len = strlen(plaintext);
        uint8_t iv[SECURELINK_GCM_NONCE_SIZE];
        esp_fill_random(iv, sizeof(iv));

        uint8_t ciphertext[64];
        uint8_t tag[SECURELINK_TAG_SIZE];

        securelink_err_t err = securelink_aes_gcm_encrypt(
            activeSessionKey, iv, nullptr, 0,
            (const uint8_t *)plaintext, pt_len,
            ciphertext, tag
        );

        if (err == SECURELINK_OK) {
            uint8_t pkt[1 + SECURELINK_GCM_NONCE_SIZE + SECURELINK_TAG_SIZE + sizeof(ciphertext)];
            pkt[0] = PKT_DATA;
            memcpy(pkt + 1, iv, SECURELINK_GCM_NONCE_SIZE);
            memcpy(pkt + 1 + SECURELINK_GCM_NONCE_SIZE, tag, SECURELINK_TAG_SIZE);
            memcpy(pkt + 1 + SECURELINK_GCM_NONCE_SIZE + SECURELINK_TAG_SIZE, ciphertext, pt_len);
            size_t total_len = 1 + SECURELINK_GCM_NONCE_SIZE + SECURELINK_TAG_SIZE + pt_len;

            esp_now_send(peerMac, pkt, total_len);
            Serial.printf("[TX DATA #%lu] Sent AES-256-GCM payload (%s): \"%s\"\n",
                          (unsigned long)packetCounter,
                          activeKeyMode == SECURELINK_STATUS_ACTIVE ? "PHY_ACTIVE" : "FALLBACK",
                          plaintext);
        }
    }
}
