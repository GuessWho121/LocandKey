#ifndef SECURELINK_TYPES_H
#define SECURELINK_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return Error Codes */
typedef enum {
    SECURELINK_OK                   = 0,
    SECURELINK_ERR_VARIANCE         = -1,
    SECURELINK_ERR_MIN_BITS         = -2,
    SECURELINK_ERR_BALANCE          = -3,
    SECURELINK_ERR_AUTH_FAILED      = -4,
    SECURELINK_ERR_REPLAY           = -5,
    SECURELINK_ERR_RECONCILE_FAILED = -6,
    SECURELINK_ERR_CONFIRM_FAILED   = -7,
    SECURELINK_ERR_INVALID_PARAM    = -8,
    SECURELINK_ERR_CRYPTO_FAILED    = -9
} securelink_err_t;

/* Session Status States */
typedef enum {
    SECURELINK_STATUS_IDLE     = 0,
    SECURELINK_STATUS_ACTIVE   = 1, /* Channel-derived key confirmed */
    SECURELINK_STATUS_FALLBACK = 2  /* Authenticated pre-shared secret fallback */
} securelink_status_t;

/* BCH(511, 457, t=6) Code Parameters */
#define SECURELINK_BCH_N              511
#define SECURELINK_BCH_K              457
#define SECURELINK_BCH_T              6
#define SECURELINK_BCH_M              9
#define SECURELINK_BCH_ECC_BITS       54
#define SECURELINK_BCH_ECC_BYTES      7
#define SECURELINK_BCH_MAX_DATA_BITS  456
#define SECURELINK_BCH_MAX_DATA_BYTES 57

/* Cryptographic Sizes (bytes) */
#define SECURELINK_NONCE_SIZE         16
#define SECURELINK_DIGEST_SIZE        48 /* SHA-384 */
#define SECURELINK_KEY_SIZE           32 /* 256-bit AES key */
#define SECURELINK_TAG_SIZE           16 /* AES-GCM tag */
#define SECURELINK_GCM_NONCE_SIZE     12 /* AES-GCM IV */

/* Channel / Subcarrier Constants */
#define SECURELINK_SUBCARRIERS        64
#define SECURELINK_MAX_ROUNDS         4
#define SECURELINK_PACKETS_PER_ROUND  8
#define SECURELINK_MAX_FEATURES       (SECURELINK_SUBCARRIERS * SECURELINK_PACKETS_PER_ROUND)
#define SECURELINK_MASK_BYTES         ((SECURELINK_MAX_FEATURES + 7) / 8)

/* Gate Threshold Defaults */
#define SECURELINK_MIN_VARIANCE_DEF   0.0001f
#define SECURELINK_GUARD_RATIO_DEF    0.25f
#define SECURELINK_MIN_BITS_DEF       256

/* Replay Nonce Cache Size */
#define SECURELINK_NONCE_CACHE_SIZE   16

/* Offer Packet Payload (Alice -> Bob) */
typedef struct {
    uint8_t nonce[SECURELINK_NONCE_SIZE];
    uint8_t transcript_digest[SECURELINK_DIGEST_SIZE];
    uint16_t feature_count;
    uint8_t reliability_mask[SECURELINK_MASK_BYTES];
    uint16_t mask_bytes;
    uint8_t helper_data[SECURELINK_BCH_ECC_BYTES];
    uint8_t auth_tag[SECURELINK_DIGEST_SIZE];
} securelink_offer_t;

/* Reply Confirmation Packet Payload (Bob -> Alice) */
typedef struct {
    uint8_t offer_digest[SECURELINK_DIGEST_SIZE];
    uint8_t confirmation_tag[SECURELINK_DIGEST_SIZE];
} securelink_reply_t;

/* Alice Session Context */
typedef struct {
    uint8_t key[SECURELINK_KEY_SIZE];
    uint8_t offer_digest[SECURELINK_DIGEST_SIZE];
    securelink_status_t status;
} securelink_alice_session_t;

/* Bob Session Context */
typedef struct {
    uint8_t key[SECURELINK_KEY_SIZE];
    securelink_status_t status;
    uint16_t corrected_bits;
    float correction_rate;
} securelink_bob_session_t;

#ifdef __cplusplus
}
#endif

#endif /* SECURELINK_TYPES_H */
