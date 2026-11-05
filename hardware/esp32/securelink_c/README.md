# SecureLink Embedded C / C++ Engine (ESP32 / ESP-IDF)

Native embedded C re-implementation of the LocandKey SecureLink physical-layer key generation protocol, optimized for 2.4 GHz IEEE 802.11n Wi-Fi IoT microcontrollers (Espressif ESP32, ESP32-S3, ESP32-C3).

---

## 1. Directory Structure

```
hardware/esp32/securelink_c/
├── include/
│   ├── securelink_types.h       # Protocol structures, constants, error codes
│   ├── securelink_bch.h         # BCH(511, 457, t=6) codec interface over GF(2^9)
│   ├── securelink_feature.h     # CSI subcarrier magnitude, variance & guard quantization
│   ├── securelink_crypto.h      # mbedTLS wrappers (HKDF-SHA-384, HMAC, AES-256-GCM)
│   └── securelink_protocol.h    # Alice & Bob state machines, offer/reply, fallback
├── src/
│   ├── securelink_bch.c         # Self-contained BCH encoder, Berlekamp-Massey & Chien
│   ├── securelink_feature.c     # Dynamic variance filter & median-normalized quantization
│   ├── securelink_crypto.c      # Hardware-accelerated cryptographic primitives
│   └── securelink_protocol.c    # End-to-end handshake, replay prevention & fallback
├── test/
│   └── test_securelink_c.c      # Comprehensive assert-based test runner
├── examples/
│   └── securelink_firmware_node/
│       └── securelink_firmware_node.ino # Arduino ESP32 firmware integration sketch
├── libsecurelink.a              # Pre-compiled static library for Xtensa ESP32
└── README.md
```

---

## 2. Technical Capabilities

- **Zero-Dependency BCH(511, 457, $t=6$) Codec**:
  - Operates over Galois Field $GF(2^9)$ with primitive polynomial $p(x) = x^9 + x^4 + 1$ (`0x211`).
  - Bit-for-bit identical to the Python Linux-kernel-based `bchlib` encoder and syndrome decoder.
  - Corrects up to 6 bit errors per block; detects $>6$ errors without silent corruption.
- **Dynamic Channel Quality Filtering**:
  - Computes temporal variance across packet bursts ($V / P$) to reject static channels lacking multipath entropy.
  - Discards noise-sensitive threshold boundaries using configurable guard ratio ($q = 0.25$ or $0.15$).
  - Enforces minimum bit count ($\ge 256$ bits) and balance constraints ($0.25 \le \text{balance} \le 0.75$).
- **Cryptographic Binding via mbedTLS**:
  - Derives context-separated 256-bit AES keys using **HKDF-SHA-384** (RFC 5869).
  - Binds session nonces, reliability masks, public helper data, and transcripts via **HMAC-SHA-384**.
  - Authenticated payload encryption and decryption via **AES-256-GCM** with 12-byte IVs.
- **Fail-Closed Fallback Engine**:
  - Automatically transitions to authenticated pre-shared secret fallback upon any channel gate failure, BCH decoding error, or confirmation mismatch.
- **Sliding-Window Replay Cache**:
  - Rejects replayed offer nonces within an active session ring buffer.

---

## 3. Building and Verification

### Compilation for Xtensa ESP32:
```powershell
$gcc = "C:\Users\aksha\AppData\Local\Arduino15\packages\esp32\tools\esp-x32\2601\bin\xtensa-esp32-elf-gcc.exe"
$inc_mbedtls = "C:\Users\aksha\AppData\Local\Arduino15\packages\esp32\tools\esp32-libs\3.3.10\include\mbedtls\mbedtls\include"
$inc_local = "hardware/esp32/securelink_c/include"

& $gcc -I $inc_local -I $inc_mbedtls -Wall -Wextra -Werror -O2 -c hardware/esp32/securelink_c/src/securelink_bch.c -o securelink_bch.o
& $gcc -I $inc_local -I $inc_mbedtls -Wall -Wextra -Werror -O2 -c hardware/esp32/securelink_c/src/securelink_feature.c -o securelink_feature.o
& $gcc -I $inc_local -I $inc_mbedtls -Wall -Wextra -Werror -O2 -c hardware/esp32/securelink_c/src/securelink_crypto.c -o securelink_crypto.o
& $gcc -I $inc_local -I $inc_mbedtls -Wall -Wextra -Werror -O2 -c hardware/esp32/securelink_c/src/securelink_protocol.c -o securelink_protocol.o
```

### Static Library Archive:
```powershell
$ar = "C:\Users\aksha\AppData\Local\Arduino15\packages\esp32\tools\esp-x32\2601\bin\xtensa-esp32-elf-ar.exe"
& $ar rcs hardware/esp32/securelink_c/libsecurelink.a securelink_bch.o securelink_feature.o securelink_crypto.o securelink_protocol.o
```

### Python Cross-Verification Check:
```powershell
uv run --with bchlib python hardware/esp32/test_securelink_c.py
```
Outputs:
```text
[PASS] BCH C-LFSR matches Python bchlib bit-for-bit.
[PASS] Offer wire-format header serialization matches C struct layout.
All cross-verification checks passed successfully!
```
