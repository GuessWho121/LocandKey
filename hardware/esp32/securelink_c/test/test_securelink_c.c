#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "securelink_types.h"
#include "securelink_bch.h"
#include "securelink_feature.h"
#include "securelink_crypto.h"
#include "securelink_protocol.h"

static void test_bch_codec(void) {
    printf("[TEST] BCH(511, 457, t=6) encode & error correction...\n");
    securelink_bch_ctx_t bch;
    securelink_bch_init(&bch);
    assert(bch.initialized);

    uint8_t original[38];
    memcpy(original, "Hello SecureLink ESP32 Physical Layer!", 38);

    uint8_t ecc[SECURELINK_BCH_ECC_BYTES];
    securelink_bch_encode(&bch, original, sizeof(original), ecc);

    /* Verify no error decode */
    uint8_t buffer[38];
    memcpy(buffer, original, sizeof(original));
    int res = securelink_bch_decode(&bch, buffer, sizeof(buffer), ecc);
    assert(res == 0);
    assert(memcmp(buffer, original, sizeof(original)) == 0);

    /* Test 1 to 6 bit errors */
    for (int num_errors = 1; num_errors <= 6; ++num_errors) {
        memcpy(buffer, original, sizeof(original));
        /* Inject bit errors at spread positions */
        for (int e = 0; e < num_errors; ++e) {
            size_t byte_pos = e * 6;
            buffer[byte_pos] ^= (1 << (e % 8));
        }
        res = securelink_bch_decode(&bch, buffer, sizeof(buffer), ecc);
        assert(res == num_errors);
        assert(memcmp(buffer, original, sizeof(original)) == 0);
    }

    /* Test uncorrectable errors (7 bit errors) */
    memcpy(buffer, original, sizeof(original));
    for (int e = 0; e < 7; ++e) {
        size_t byte_pos = e * 5;
        buffer[byte_pos] ^= (1 << (e % 8));
    }
    res = securelink_bch_decode(&bch, buffer, sizeof(buffer), ecc);
    assert(res < 0); /* Must fail */

    printf("  [PASS] BCH encode & decode verified up to t=6.\n");
}

static void test_crypto_primitives(void) {
    printf("[TEST] Cryptographic primitives (HKDF, HMAC, AES-GCM)...\n");

    /* HKDF test */
    uint8_t ikm[32] = {0x01, 0x02, 0x03};
    uint8_t salt[16] = {0x10, 0x20};
    uint8_t info[] = "test/info";
    uint8_t key1[32], key2[32];
    assert(securelink_hkdf_sha384(ikm, sizeof(ikm), salt, sizeof(salt), info, sizeof(info)-1, key1, sizeof(key1)) == SECURELINK_OK);
    assert(securelink_hkdf_sha384(ikm, sizeof(ikm), salt, sizeof(salt), info, sizeof(info)-1, key2, sizeof(key2)) == SECURELINK_OK);
    assert(memcmp(key1, key2, 32) == 0);

    /* HMAC test */
    uint8_t tag1[SECURELINK_DIGEST_SIZE], tag2[SECURELINK_DIGEST_SIZE];
    assert(securelink_hmac_sha384(key1, 32, (const uint8_t*)"message", 7, tag1) == SECURELINK_OK);
    assert(securelink_hmac_sha384(key1, 32, (const uint8_t*)"message", 7, tag2) == SECURELINK_OK);
    assert(memcmp(tag1, tag2, SECURELINK_DIGEST_SIZE) == 0);

    /* AES-256-GCM authenticated encryption */
    uint8_t iv[SECURELINK_GCM_NONCE_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    uint8_t aad[] = "sensor-header";
    uint8_t plaintext[] = "temperature: 24.5C, humidity: 55%";
    size_t pt_len = sizeof(plaintext);
    uint8_t ciphertext[sizeof(plaintext)];
    uint8_t gcm_tag[SECURELINK_TAG_SIZE];
    uint8_t decrypted[sizeof(plaintext)];

    assert(securelink_aes_gcm_encrypt(key1, iv, aad, sizeof(aad)-1, plaintext, pt_len, ciphertext, gcm_tag) == SECURELINK_OK);
    assert(securelink_aes_gcm_decrypt(key1, iv, aad, sizeof(aad)-1, ciphertext, pt_len, gcm_tag, decrypted) == SECURELINK_OK);
    assert(memcmp(plaintext, decrypted, pt_len) == 0);

    /* Tamper test: modify ciphertext */
    ciphertext[0] ^= 0x01;
    assert(securelink_aes_gcm_decrypt(key1, iv, aad, sizeof(aad)-1, ciphertext, pt_len, gcm_tag, decrypted) != SECURELINK_OK);

    printf("  [PASS] Crypto primitives verified.\n");
}

static void test_end_to_end_protocol(void) {
    printf("[TEST] End-to-end SecureLink Alice/Bob handshake...\n");
    securelink_bch_ctx_t bch;
    securelink_bch_init(&bch);

    securelink_replay_cache_t replay_cache;
    securelink_replay_cache_init(&replay_cache);

    uint8_t auth_secret[SECURELINK_KEY_SIZE];
    memset(auth_secret, 0x5a, SECURELINK_KEY_SIZE);

    uint8_t nonce[SECURELINK_NONCE_SIZE] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                                            0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10};
    uint8_t transcript[] = "session-handshake-v1";

    /* Generate synthetic dynamic multi-frame CSI for Alice and Bob with minor noise */
    size_t frames = 8;
    size_t subcarriers = 320; /* 5 rounds of 64 = 320 features */
    int8_t alice_csi[8 * 320 * 2];
    int8_t bob_csi[8 * 320 * 2];

    for (size_t f = 0; f < frames; ++f) {
        for (size_t k = 0; k < subcarriers; ++k) {
            size_t idx = (f * subcarriers + k) * 2;
            /* Underlying dynamic channel */
            float phase = 2.0f * 3.14159f * ((float)k / (float)subcarriers) + 0.3f * (float)f;
            int8_t r = (int8_t)(35.0f * cosf(phase));
            int8_t i = (int8_t)(35.0f * sinf(phase));

            alice_csi[idx] = r;
            alice_csi[idx + 1] = i;

            /* Bob has identical channel plus slight measurement variation (reciprocal) */
            bob_csi[idx] = (k % 40 == 0) ? (r + 2) : r;
            bob_csi[idx + 1] = i;
        }
    }

    /* 1. Alice creates offer */
    securelink_offer_t offer;
    securelink_alice_session_t alice_session;
    securelink_err_t err = securelink_alice_create_offer(
        alice_csi, frames, subcarriers, auth_secret, nonce, transcript, sizeof(transcript)-1,
        &bch, &offer, &alice_session
    );
    assert(err == SECURELINK_OK);
    assert(alice_session.status == SECURELINK_STATUS_IDLE);

    /* 2. Bob accepts offer and generates confirmation reply */
    securelink_reply_t reply;
    securelink_bob_session_t bob_session;
    err = securelink_bob_accept_offer(
        bob_csi, frames, subcarriers, &offer, auth_secret, nonce, transcript, sizeof(transcript)-1,
        &bch, &replay_cache, &reply, &bob_session
    );
    assert(err == SECURELINK_OK);
    assert(bob_session.status == SECURELINK_STATUS_ACTIVE);

    /* 3. Alice finalizes session */
    err = securelink_alice_finalize(&alice_session, &reply);
    assert(err == SECURELINK_OK);
    assert(alice_session.status == SECURELINK_STATUS_ACTIVE);

    /* 4. Verify Alice and Bob derived identical 256-bit keys */
    assert(memcmp(alice_session.key, bob_session.key, SECURELINK_KEY_SIZE) == 0);

    /* 5. Negative Security Tests: Replay Attack */
    err = securelink_bob_accept_offer(
        bob_csi, frames, subcarriers, &offer, auth_secret, nonce, transcript, sizeof(transcript)-1,
        &bch, &replay_cache, &reply, &bob_session
    );
    assert(err == SECURELINK_ERR_REPLAY);

    /* 6. Negative Security Tests: Tampered Offer */
    offer.auth_tag[0] ^= 0xff;
    err = securelink_bob_accept_offer(
        bob_csi, frames, subcarriers, &offer, auth_secret, nonce, transcript, sizeof(transcript)-1,
        &bch, NULL, &reply, &bob_session
    );
    assert(err == SECURELINK_ERR_AUTH_FAILED);

    /* 7. Negative Security Tests: Static Channel Fallback */
    int8_t static_csi[8 * 320 * 2];
    for (size_t idx = 0; idx < sizeof(static_csi); ++idx) {
        static_csi[idx] = (int8_t)(idx % 50);
    }
    uint8_t nonce2[SECURELINK_NONCE_SIZE] = {0x99};
    err = securelink_alice_create_offer(
        static_csi, frames, subcarriers, auth_secret, nonce2, transcript, sizeof(transcript)-1,
        &bch, &offer, &alice_session
    );
    assert(err == SECURELINK_ERR_VARIANCE);
    assert(alice_session.status == SECURELINK_STATUS_FALLBACK);

    printf("  [PASS] Full handshake, key agreement, replay rejection, and fallback verified!\n");
}

int main(void) {
    printf("=========================================\n");
    printf("SecureLink Embedded C Engine Verification\n");
    printf("=========================================\n");
    test_bch_codec();
    test_crypto_primitives();
    test_end_to_end_protocol();
    printf("=========================================\n");
    printf("ALL EMBEDDED C TESTS PASSED SUCCESSFULLY!\n");
    printf("=========================================\n");
    return 0;
}
