#include <Arduino.h>
#include <Preferences.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <mbedtls/hkdf.h>

constexpr uint8_t channel = 6;
constexpr uint8_t aliceMac[] = {0x38, 0x18, 0x2b, 0xb2, 0x87, 0x68};
constexpr uint8_t bobMac[] = {0x2c, 0xbc, 0xbb, 0x0c, 0x67, 0xe4};
constexpr uint8_t broadcastMac[] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
constexpr uint32_t probeMagic = 0x4c4b4353;
constexpr uint32_t telemetryMagic = 0x4c4b4354;
constexpr uint32_t probeIntervalMs = 100;
constexpr size_t csiBytes = 128;

struct Probe {
  uint32_t magic;
  uint32_t captureId;
  uint32_t sequence;
  uint32_t sentAtUs;
};

struct Sample {
  uint32_t captureId;
  uint32_t sequence;
  uint32_t timestampUs;
  uint32_t roundTripUs;
  int8_t rssi;
  int8_t noiseFloor;
  uint8_t channel;
  uint8_t signalMode;
  uint8_t rate;
  uint8_t mcs;
  uint8_t channelWidth;
  uint8_t secondaryChannel;
  uint8_t aggregation;
  uint8_t antenna;
  uint8_t rxState;
  bool firstWordInvalid;
  uint16_t csiLength;
  int8_t csi[csiBytes];
};

struct Telemetry {
  uint32_t magic;
  Sample sample;
  uint32_t probesSeen;
  uint32_t csiCaptured;
  uint32_t repliesQueued;
  uint32_t relayAttempts;
  uint32_t queueDrops;
};

static_assert(sizeof(Telemetry) <= ESP_NOW_MAX_DATA_LEN, "CSI telemetry exceeds ESP-NOW v1 payload");

constexpr bool validTelemetryHeader(uint32_t magic, uint32_t receivedCapture,
                                    uint16_t length, uint8_t receivedChannel, uint32_t expectedCapture) {
  return magic == telemetryMagic && receivedCapture == expectedCapture &&
         length == csiBytes && receivedChannel == channel;
}

constexpr bool freshTelemetrySequence(uint32_t sequence, uint32_t previous, bool havePrevious) {
  return !havePrevious || sequence > previous;
}

static_assert(validTelemetryHeader(telemetryMagic, 7, csiBytes, channel, 7), "Valid telemetry rejected");
static_assert(!validTelemetryHeader(telemetryMagic, 8, csiBytes, channel, 7), "Wrong capture accepted");
static_assert(!validTelemetryHeader(telemetryMagic, 7, csiBytes - 1, channel, 7), "Malformed CSI accepted");
static_assert(!freshTelemetrySequence(4, 4, true), "Duplicate telemetry accepted");

QueueHandle_t samples;
QueueHandle_t replies;
QueueHandle_t relaySamples;
QueueHandle_t remoteSamples;
bool isAlice;
const uint8_t *peerMac;
uint32_t captureId;
uint32_t nextSequence;
uint32_t lastProbeMs;
uint32_t lastRemoteSequence;
bool haveRemoteSequence;
volatile uint32_t relayRejected;
volatile uint32_t relayDuplicates;
volatile uint32_t relayQueueDrops;
volatile uint32_t probesSeen;
volatile uint32_t csiCaptured;
volatile uint32_t repliesQueued;
volatile uint32_t relayAttempts;
volatile uint32_t probeQueued;
volatile uint32_t remoteAccepted;
volatile uint32_t localQueueDrops;
uint32_t localPrinted;
uint32_t bobPrinted;
Telemetry lastTelemetry;
Preferences storage;

void fail(const char *operation, esp_err_t error) {
  if (error != ESP_OK) {
    Serial.printf("# %s failed: %s\n", operation, esp_err_to_name(error));
    while (true) delay(1000);
  }
}

void onCsi(void *, wifi_csi_info_t *info) {
  if (!info || !info->buf || !info->payload || info->len < csiBytes ||
      info->payload_len < 15 + sizeof(Probe) || memcmp(info->mac, peerMac, 6) != 0) {
    return;
  }
  Probe probe;
  memcpy(&probe, info->payload + 15, sizeof(probe));
  if (probe.magic != probeMagic) return;
  Sample sample = {};
  sample.captureId = probe.captureId;
  sample.sequence = probe.sequence;
  sample.timestampUs = info->rx_ctrl.timestamp;
  sample.roundTripUs = isAlice ? uint32_t(micros() - probe.sentAtUs) : 0;
  sample.rssi = info->rx_ctrl.rssi;
  sample.noiseFloor = info->rx_ctrl.noise_floor;
  sample.channel = info->rx_ctrl.channel;
  sample.signalMode = info->rx_ctrl.sig_mode;
  sample.rate = info->rx_ctrl.rate;
  sample.mcs = info->rx_ctrl.mcs;
  sample.channelWidth = info->rx_ctrl.cwb;
  sample.secondaryChannel = info->rx_ctrl.secondary_channel;
  sample.aggregation = info->rx_ctrl.aggregation;
  sample.antenna = info->rx_ctrl.ant;
  sample.rxState = info->rx_ctrl.rx_state;
  sample.firstWordInvalid = info->first_word_invalid;
  sample.csiLength = info->len;
  memcpy(sample.csi, info->buf, csiBytes);
  if (info->first_word_invalid) memset(sample.csi, 0, 4);
  ++csiCaptured;
  if (xQueueSend(samples, &sample, 0) != pdTRUE) ++localQueueDrops;
  if (!isAlice && xQueueSend(relaySamples, &sample, 0) != pdTRUE) ++relayQueueDrops;
}

void onReceive(const esp_now_recv_info_t *info, const uint8_t *data, int length) {
  if (!info || !data || memcmp(info->src_addr, peerMac, 6) != 0) return;
  if (isAlice) {
    if (length == sizeof(Probe)) return;
    if (length != sizeof(Telemetry)) {
      ++relayRejected;
      return;
    }
    Telemetry telemetry;
    memcpy(&telemetry, data, sizeof(telemetry));
    const Sample &sample = telemetry.sample;
    if (!validTelemetryHeader(telemetry.magic, sample.captureId, sample.csiLength,
                              sample.channel, captureId)) {
      ++relayRejected;
      return;
    }
    if (!freshTelemetrySequence(sample.sequence, lastRemoteSequence, haveRemoteSequence)) {
      ++relayDuplicates;
      return;
    }
    if (xQueueSend(remoteSamples, &sample, 0) != pdTRUE) {
      ++relayQueueDrops;
      return;
    }
    lastRemoteSequence = sample.sequence;
    haveRemoteSequence = true;
    lastTelemetry = telemetry;
    ++remoteAccepted;
    return;
  }
  if (length != sizeof(Probe)) return;
  Probe probe;
  memcpy(&probe, data, sizeof(probe));
  if (probe.magic == probeMagic) {
    ++probesSeen;
    if (xQueueSend(replies, &probe, 0) != pdTRUE) ++relayQueueDrops;
  }
}

void sendReplies(void *) {
  Probe reply;
  while (true) {
    if (xQueueReceive(replies, &reply, portMAX_DELAY) == pdTRUE) {
      esp_err_t error = esp_now_send(broadcastMac, reinterpret_cast<uint8_t *>(&reply), sizeof(reply));
      if (error == ESP_OK) ++repliesQueued;
      if (error != ESP_OK) Serial.printf("# reply failed: %s\n", esp_err_to_name(error));
      Sample sample;
      while (xQueueReceive(relaySamples, &sample, pdMS_TO_TICKS(10)) == pdTRUE) {
        if (sample.sequence != reply.sequence || sample.captureId != reply.captureId) continue;
        vTaskDelay(pdMS_TO_TICKS(5));
        ++relayAttempts;
        Telemetry telemetry = {telemetryMagic, sample, probesSeen, csiCaptured,
                               repliesQueued, relayAttempts, relayQueueDrops};
        error = esp_now_send(peerMac, reinterpret_cast<uint8_t *>(&telemetry), sizeof(telemetry));
        if (error != ESP_OK) Serial.printf("# telemetry failed: %s\n", esp_err_to_name(error));
        break;
      }
    }
  }
}

void printSample(const Sample &sample, const char *device) {
  Serial.printf("{\"device\":\"%s\",\"capture_id\":%lu,\"sequence\":%lu,\"timestamp_us\":%lu,\"round_trip_us\":%lu,\"rssi_dbm\":%d,\"noise_floor_dbm\":%d,\"channel\":%u,\"bandwidth_mhz\":%u,\"signal_mode\":%u,\"rate\":%u,\"mcs\":%u,\"secondary_channel\":%u,\"aggregation\":%u,\"antenna\":%u,\"rx_state\":%u,\"first_word_invalid\":%u,\"csi_length\":%u,\"csi\":[",
                device, (unsigned long)sample.captureId, (unsigned long)sample.sequence,
                (unsigned long)sample.timestampUs, (unsigned long)sample.roundTripUs,
                sample.rssi, sample.noiseFloor, sample.channel, sample.channelWidth ? 40 : 20,
                sample.signalMode, sample.rate, sample.mcs, sample.secondaryChannel,
                sample.aggregation, sample.antenna, sample.rxState, sample.firstWordInvalid,
                sample.csiLength);
  for (size_t index = 0; index < csiBytes; ++index) {
    if (index) Serial.print(',');
    Serial.print(sample.csi[index]);
  }
  Serial.println("]}");
}

void setup() {
  Serial.begin(115200);
  delay(500);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  uint8_t localMac[6];
  fail("get MAC", esp_wifi_get_mac(WIFI_IF_STA, localMac));
  isAlice = memcmp(localMac, aliceMac, 6) == 0;
  peerMac = isAlice ? bobMac : aliceMac;
  if (!isAlice && memcmp(localMac, bobMac, 6) != 0) {
    Serial.println("# Unknown board MAC; refusing to send probes");
    while (true) delay(1000);
  }
  if (!storage.begin("securelink", true) || storage.getBytesLength("secret") != 32) {
    Serial.println("# Provisioned pair secret required for CSI telemetry");
    while (true) delay(1000);
  }
  uint8_t secret[32];
  if (storage.getBytes("secret", secret, sizeof(secret)) != sizeof(secret)) {
    Serial.println("# Pair secret read failed");
    while (true) delay(1000);
  }
  storage.end();
  uint8_t keys[32];
  const char context[] = "securelink/csi-telemetry/v1";
  const mbedtls_md_info_t *hash = mbedtls_md_info_from_type(MBEDTLS_MD_SHA384);
  if (!hash || mbedtls_hkdf(hash, nullptr, 0, secret, sizeof(secret),
      reinterpret_cast<const uint8_t *>(context), sizeof(context) - 1, keys, sizeof(keys)) != 0) {
    Serial.println("# CSI telemetry key derivation failed");
    while (true) delay(1000);
  }
  memset(secret, 0, sizeof(secret));
  captureId = isAlice ? esp_random() : 0;
  samples = xQueueCreate(32, sizeof(Sample));
  replies = xQueueCreate(16, sizeof(Probe));
  relaySamples = xQueueCreate(32, sizeof(Sample));
  remoteSamples = xQueueCreate(32, sizeof(Sample));
  if (!samples || !replies || !relaySamples || !remoteSamples) {
    Serial.println("# Queue allocation failed");
    while (true) delay(1000);
  }
  if (!isAlice && xTaskCreate(sendReplies, "esp_now_reply", 4096, nullptr, 2, nullptr) != pdPASS) {
    Serial.println("# Reply task creation failed");
    while (true) delay(1000);
  }
  fail("set channel", esp_wifi_set_channel(channel, WIFI_SECOND_CHAN_NONE));
  fail("set bandwidth", esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));
  fail("ESP-NOW init", esp_now_init());
  fail("set telemetry PMK", esp_now_set_pmk(keys));
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, broadcastMac, 6);
  peer.channel = channel;
  peer.ifidx = WIFI_IF_STA;
  fail("add peer", esp_now_add_peer(&peer));
  peer = {};
  memcpy(peer.peer_addr, peerMac, 6);
  memcpy(peer.lmk, keys + 16, 16);
  peer.channel = channel;
  peer.ifidx = WIFI_IF_STA;
  peer.encrypt = true;
  fail("add telemetry peer", esp_now_add_peer(&peer));
  memset(keys, 0, sizeof(keys));
  esp_now_rate_config_t rate = {};
  rate.phymode = WIFI_PHY_MODE_HT20;
  rate.rate = WIFI_PHY_RATE_MCS0_LGI;
  fail("set probe rate", esp_now_set_peer_rate_config(broadcastMac, &rate));
  fail("register receiver", esp_now_register_recv_cb(onReceive));
  fail("promiscuous mode", esp_wifi_set_promiscuous(true));
  wifi_csi_config_t config = {};
  config.lltf_en = false;
  config.htltf_en = true;
  config.stbc_htltf2_en = false;
  config.ltf_merge_en = false;
  config.channel_filter_en = true;
  fail("CSI config", esp_wifi_set_csi_config(&config));
  fail("CSI callback", esp_wifi_set_csi_rx_cb(onCsi, nullptr));
  fail("CSI enable", esp_wifi_set_csi(true));
  Serial.printf("# SecureLink CSI probe ready: %s, channel %u, 115200 baud\n",
                isAlice ? "alice" : "bob", channel);
}

void loop() {
  Sample sample;
  while (xQueueReceive(samples, &sample, 0) == pdTRUE) {
    printSample(sample, isAlice ? "alice" : "bob");
    ++localPrinted;
  }
  if (isAlice) {
    while (xQueueReceive(remoteSamples, &sample, 0) == pdTRUE) {
      printSample(sample, "bob");
      ++bobPrinted;
    }
  }
  if (isAlice && millis() - lastProbeMs >= probeIntervalMs) {
    lastProbeMs = millis();
    Probe probe = {probeMagic, captureId, nextSequence++, micros()};
    esp_err_t error = esp_now_send(broadcastMac, reinterpret_cast<uint8_t *>(&probe), sizeof(probe));
    if (error == ESP_OK) ++probeQueued;
    if (error != ESP_OK) Serial.printf("# probe failed: %s\n", esp_err_to_name(error));
  }
  static uint32_t lastStatsMs;
  if (millis() - lastStatsMs >= 5000) {
    lastStatsMs = millis();
    if (isAlice) Serial.printf("# relay_stats rejected=%lu duplicates=%lu queue_drops=%lu\n",
                                (unsigned long)relayRejected, (unsigned long)relayDuplicates,
                                (unsigned long)relayQueueDrops);
    Serial.printf("# %s_stats probe_ok=%lu probes_seen=%lu csi=%lu reply_ok=%lu relay_attempts=%lu local_printed=%lu bob_printed=%lu remote_accepted=%lu queue_drops=%lu local_drops=%lu\n",
                  isAlice ? "alice" : "bob", (unsigned long)probeQueued,
                  (unsigned long)probesSeen, (unsigned long)csiCaptured,
                  (unsigned long)repliesQueued, (unsigned long)relayAttempts,
                  (unsigned long)localPrinted, (unsigned long)bobPrinted,
                  (unsigned long)remoteAccepted, (unsigned long)relayQueueDrops,
                  (unsigned long)localQueueDrops);
    if (isAlice && remoteAccepted) {
      Serial.printf("# bob_relay_stats probes_seen=%lu csi=%lu reply_ok=%lu relay_attempts=%lu queue_drops=%lu\n",
                    (unsigned long)lastTelemetry.probesSeen, (unsigned long)lastTelemetry.csiCaptured,
                    (unsigned long)lastTelemetry.repliesQueued, (unsigned long)lastTelemetry.relayAttempts,
                    (unsigned long)lastTelemetry.queueDrops);
    }
  }
  delay(1);
}
