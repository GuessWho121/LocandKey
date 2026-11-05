#include <Arduino.h>
#include <Preferences.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <mbedtls/gcm.h>
#include <mbedtls/hkdf.h>
#include <mbedtls/md.h>

constexpr uint8_t channel = 6;
constexpr uint8_t aliceMac[] = {0x38, 0x18, 0x2b, 0xb2, 0x87, 0x68};
constexpr uint8_t bobMac[] = {0x2c, 0xbc, 0xbb, 0x0c, 0x67, 0xe4};
constexpr uint8_t frameVersion = 1;
constexpr uint32_t sendIntervalMs = 500;
constexpr int sendCount = 120;

struct __attribute__((packed)) Frame {
  uint8_t version;
  uint32_t boot;
  uint64_t sequence;
  uint8_t ciphertext[4];
  uint8_t tag[16];
};

struct ReplayState {
  uint32_t boot;
  uint64_t sequence;
};

Preferences storage;
mbedtls_gcm_context cipher;
QueueHandle_t incoming;
const uint8_t *peerMac;
bool isAlice;
uint32_t bootEpoch;
uint64_t nextSequence = 1;
ReplayState replayState = {};
Frame lastFrame = {};
bool haveLastFrame = false;
int remaining = 0;
uint32_t lastSendMs = 0;

void fail(const char *stage) {
  Serial.printf("ERROR %s\n", stage);
  while (true) delay(1000);
}

void requireEsp(esp_err_t result, const char *stage) {
  if (result != ESP_OK) fail(stage);
}

int hexValue(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  return -1;
}

void readSecret(uint8_t secret[32]) {
  size_t length = storage.getBytesLength("secret");
  if (length == 32) {
    if (storage.getBytes("secret", secret, 32) != 32) fail("secret_read");
    return;
  }
  if (length != 0) fail("secret_size");
  Serial.println("PROVISION_REQUIRED");
  while (true) {
    if (!Serial.available()) {
      delay(20);
      continue;
    }
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() != 64) {
      Serial.println("PROVISION_INVALID");
      continue;
    }
    bool valid = true;
    for (size_t index = 0; index < 32; ++index) {
      int high = hexValue(line[2 * index]);
      int low = hexValue(line[2 * index + 1]);
      if (high < 0 || low < 0) {
        valid = false;
        break;
      }
      secret[index] = uint8_t((high << 4) | low);
    }
    if (!valid) {
      Serial.println("PROVISION_INVALID");
      continue;
    }
    if (storage.putBytes("secret", secret, 32) != 32) fail("secret_write");
    Serial.println("PROVISIONED");
    return;
  }
}

void makeNonce(const Frame &frame, uint8_t nonce[12]) {
  memcpy(nonce, &frame.boot, sizeof(frame.boot));
  memcpy(nonce + sizeof(frame.boot), &frame.sequence, sizeof(frame.sequence));
}

void onReceive(const esp_now_recv_info_t *info, const uint8_t *data, int length) {
  if (isAlice || !info || !data || length != sizeof(Frame) ||
      memcmp(info->src_addr, peerMac, 6) != 0) return;
  xQueueSend(incoming, data, 0);
}

void sendWireFrame(const Frame &frame, const char *kind) {
  esp_err_t result = esp_now_send(peerMac, reinterpret_cast<const uint8_t *>(&frame), sizeof(frame));
  Serial.printf("TX %s boot=%lu seq=%llu queued=%d\n", kind, (unsigned long)frame.boot,
                (unsigned long long)frame.sequence, result == ESP_OK);
}

void sendReading() {
  if (nextSequence == UINT64_MAX) fail("sequence_exhausted");
  Frame frame = {};
  frame.version = frameVersion;
  frame.boot = bootEpoch;
  frame.sequence = nextSequence++;
  uint32_t uptimeMs = millis();
  uint8_t nonce[12];
  makeNonce(frame, nonce);
  int result = mbedtls_gcm_crypt_and_tag(&cipher, MBEDTLS_GCM_ENCRYPT, sizeof(uptimeMs), nonce,
      sizeof(nonce), reinterpret_cast<const uint8_t *>(&frame), offsetof(Frame, ciphertext),
      reinterpret_cast<const uint8_t *>(&uptimeMs), frame.ciphertext, sizeof(frame.tag), frame.tag);
  if (result != 0) fail("encrypt");
  sendWireFrame(frame, "reading");
  lastFrame = frame;
  haveLastFrame = true;
}

void receiveFrame(const Frame &frame) {
  if (frame.version != frameVersion) {
    Serial.println("REJECT format");
    return;
  }
  uint8_t nonce[12];
  makeNonce(frame, nonce);
  uint32_t uptimeMs = 0;
  int result = mbedtls_gcm_auth_decrypt(&cipher, sizeof(uptimeMs), nonce, sizeof(nonce),
      reinterpret_cast<const uint8_t *>(&frame), offsetof(Frame, ciphertext), frame.tag,
      sizeof(frame.tag), frame.ciphertext, reinterpret_cast<uint8_t *>(&uptimeMs));
  if (result != 0) {
    Serial.printf("REJECT tamper boot=%lu seq=%llu\n", (unsigned long)frame.boot,
                  (unsigned long long)frame.sequence);
    return;
  }
  if (frame.boot < replayState.boot ||
      (frame.boot == replayState.boot && frame.sequence <= replayState.sequence)) {
    Serial.printf("REJECT replay boot=%lu seq=%llu\n", (unsigned long)frame.boot,
                  (unsigned long long)frame.sequence);
    return;
  }
  ReplayState updated = {frame.boot, frame.sequence};
  if (storage.putBytes("replay", &updated, sizeof(updated)) != sizeof(updated)) fail("replay_write");
  replayState = updated;
  Serial.printf("ACCEPT boot=%lu seq=%llu uptime_ms=%lu\n", (unsigned long)frame.boot,
                (unsigned long long)frame.sequence, (unsigned long)uptimeMs);
}

void command(const String &input) {
  if (input == "reboot") {
    Serial.println("REBOOTING");
    Serial.flush();
    ESP.restart();
  } else if (input.startsWith("wifi\t")) {
    int separator = input.indexOf('\t', 5);
    String ssid = separator < 0 ? "" : input.substring(5, separator);
    String password = separator < 0 ? "" : input.substring(separator + 1);
    if (ssid.isEmpty() || ssid.length() > 32 || password.length() < 8 || password.length() > 63) {
      Serial.println("WIFI_FAILED credentials");
      return;
    }
    remaining = 0;
    WiFi.begin(ssid.c_str(), password.c_str());
    uint32_t started = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - started < 20000) delay(100);
    if (WiFi.status() != WL_CONNECTED) {
      WiFi.disconnect();
      requireEsp(esp_wifi_set_channel(channel, WIFI_SECOND_CHAN_NONE), "channel_restore");
      Serial.println("WIFI_FAILED connect");
      return;
    }
    esp_now_peer_info_t peer = {};
    requireEsp(esp_now_get_peer(peerMac, &peer), "peer_get");
    peer.channel = 0;
    requireEsp(esp_now_mod_peer(&peer), "peer_channel");
    Serial.printf("WIFI_CONNECTED role=%s channel=%u ip=%s\n", isAlice ? "alice" : "bob",
                  WiFi.channel(), WiFi.localIP().toString().c_str());
  } else if (input == "start" && isAlice) {
    remaining = sendCount;
    lastSendMs = millis() - sendIntervalMs;
    Serial.printf("START count=%d\n", sendCount);
  } else if (input == "once" && isAlice) {
    sendReading();
  } else if (input == "replay" && isAlice && haveLastFrame) {
    sendWireFrame(lastFrame, "replay");
  } else if (input == "tamper" && isAlice && haveLastFrame) {
    Frame altered = lastFrame;
    altered.tag[0] ^= 1;
    sendWireFrame(altered, "tamper");
  } else if (input == "status") {
    Serial.printf("STATUS role=%s boot=%lu last_seq=%llu remaining=%d wifi=%u channel=%u\n",
                   isAlice ? "alice" : "bob", (unsigned long)(isAlice ? bootEpoch : replayState.boot),
                   (unsigned long long)(isAlice ? nextSequence - 1 : replayState.sequence), remaining,
                   WiFi.status() == WL_CONNECTED, WiFi.channel());
  } else {
    Serial.println("ERROR command");
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1000);
  delay(500);
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  uint8_t localMac[6];
  requireEsp(esp_wifi_get_mac(WIFI_IF_STA, localMac), "mac");
  isAlice = memcmp(localMac, aliceMac, 6) == 0;
  if (!isAlice && memcmp(localMac, bobMac, 6) != 0) fail("unknown_board");
  peerMac = isAlice ? bobMac : aliceMac;
  if (!storage.begin("securelink", false)) fail("nvs_open");
  uint8_t secret[32];
  readSecret(secret);
  uint8_t keys[64];
  const char context[] = "securelink/fixed-link/v1";
  const mbedtls_md_info_t *hash = mbedtls_md_info_from_type(MBEDTLS_MD_SHA384);
  if (!hash || mbedtls_hkdf(hash, nullptr, 0, secret, sizeof(secret),
      reinterpret_cast<const uint8_t *>(context), sizeof(context) - 1, keys, sizeof(keys)) != 0) {
    fail("key_derivation");
  }
  memset(secret, 0, sizeof(secret));
  mbedtls_gcm_init(&cipher);
  if (mbedtls_gcm_setkey(&cipher, MBEDTLS_CIPHER_ID_AES, keys + 32, 256) != 0) fail("gcm_key");
  if (isAlice) {
    uint32_t prior = storage.getULong("boot", 0);
    if (prior == UINT32_MAX) fail("boot_exhausted");
    bootEpoch = prior + 1;
    if (storage.putULong("boot", bootEpoch) != sizeof(bootEpoch)) fail("boot_write");
  } else {
    size_t length = storage.getBytesLength("replay");
    if (length && (length != sizeof(replayState) ||
        storage.getBytes("replay", &replayState, sizeof(replayState)) != sizeof(replayState))) {
      fail("replay_read");
    }
  }
  requireEsp(esp_wifi_set_channel(channel, WIFI_SECOND_CHAN_NONE), "channel");
  requireEsp(esp_now_init(), "esp_now_init");
  requireEsp(esp_now_set_pmk(keys), "pmk");
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, peerMac, 6);
  memcpy(peer.lmk, keys + 16, 16);
  peer.channel = channel;
  peer.ifidx = WIFI_IF_STA;
  peer.encrypt = true;
  requireEsp(esp_now_add_peer(&peer), "peer");
  memset(keys, 0, sizeof(keys));
  incoming = xQueueCreate(16, sizeof(Frame));
  if (!incoming) fail("queue");
  requireEsp(esp_now_register_recv_cb(onReceive), "receiver");
  Serial.printf("READY role=%s boot=%lu\n", isAlice ? "alice" : "bob",
                (unsigned long)(isAlice ? bootEpoch : replayState.boot));
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    command(line);
  }
  if (isAlice && remaining > 0 && millis() - lastSendMs >= sendIntervalMs) {
    lastSendMs = millis();
    sendReading();
    --remaining;
    if (!remaining) Serial.println("DONE sending");
  }
  Frame frame;
  while (!isAlice && xQueueReceive(incoming, &frame, 0) == pdTRUE) receiveFrame(frame);
  delay(1);
}
