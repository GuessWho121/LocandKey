#ifndef SECURELINK_CRYPTO_H
#define SECURELINK_CRYPTO_H

#include "securelink_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* HKDF-SHA-384 key derivation (RFC 5869) */
securelink_err_t securelink_hkdf_sha384(
    const uint8_t *ikm, size_t ikm_len,
    const uint8_t *salt, size_t salt_len,
    const uint8_t *info, size_t info_len,
    uint8_t *okm, size_t okm_len
);

/* HMAC-SHA-384 authentication tag calculation */
securelink_err_t securelink_hmac_sha384(
    const uint8_t *key, size_t key_len,
    const uint8_t *data, size_t data_len,
    uint8_t out_tag[SECURELINK_DIGEST_SIZE]
);

/* SHA-384 digest calculation */
securelink_err_t securelink_sha384(
    const uint8_t *data, size_t data_len,
    uint8_t out_digest[SECURELINK_DIGEST_SIZE]
);

/* Constant-time memory comparison; returns true if buffers match */
bool securelink_constant_time_equal(
    const uint8_t *a,
    const uint8_t *b,
    size_t len
);

/* AES-256-GCM authenticated payload encryption */
securelink_err_t securelink_aes_gcm_encrypt(
    const uint8_t key[SECURELINK_KEY_SIZE],
    const uint8_t iv[SECURELINK_GCM_NONCE_SIZE],
    const uint8_t *aad, size_t aad_len,
    const uint8_t *plaintext, size_t plaintext_len,
    uint8_t *ciphertext_out,
    uint8_t tag_out[SECURELINK_TAG_SIZE]
);

/* AES-256-GCM authenticated payload decryption */
securelink_err_t securelink_aes_gcm_decrypt(
    const uint8_t key[SECURELINK_KEY_SIZE],
    const uint8_t iv[SECURELINK_GCM_NONCE_SIZE],
    const uint8_t *aad, size_t aad_len,
    const uint8_t *ciphertext, size_t ciphertext_len,
    const uint8_t tag[SECURELINK_TAG_SIZE],
    uint8_t *plaintext_out
);

#ifdef __cplusplus
}
#endif

#endif /* SECURELINK_CRYPTO_H */
