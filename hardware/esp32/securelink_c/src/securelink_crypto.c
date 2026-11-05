#include "securelink_crypto.h"
#include <mbedtls/hkdf.h>
#include <mbedtls/md.h>
#include <mbedtls/gcm.h>
#include <string.h>

securelink_err_t securelink_hkdf_sha384(
    const uint8_t *ikm, size_t ikm_len,
    const uint8_t *salt, size_t salt_len,
    const uint8_t *info, size_t info_len,
    uint8_t *okm, size_t okm_len
) {
    if (!ikm || !salt || !info || !okm || okm_len == 0) {
        return SECURELINK_ERR_INVALID_PARAM;
    }
    const mbedtls_md_info_t *md = mbedtls_md_info_from_type(MBEDTLS_MD_SHA384);
    if (!md) {
        return SECURELINK_ERR_CRYPTO_FAILED;
    }
    int ret = mbedtls_hkdf(md, salt, salt_len, ikm, ikm_len, info, info_len, okm, okm_len);
    return (ret == 0) ? SECURELINK_OK : SECURELINK_ERR_CRYPTO_FAILED;
}

securelink_err_t securelink_hmac_sha384(
    const uint8_t *key, size_t key_len,
    const uint8_t *data, size_t data_len,
    uint8_t out_tag[SECURELINK_DIGEST_SIZE]
) {
    if (!key || !data || !out_tag) {
        return SECURELINK_ERR_INVALID_PARAM;
    }
    const mbedtls_md_info_t *md = mbedtls_md_info_from_type(MBEDTLS_MD_SHA384);
    if (!md) {
        return SECURELINK_ERR_CRYPTO_FAILED;
    }
    int ret = mbedtls_md_hmac(md, key, key_len, data, data_len, out_tag);
    return (ret == 0) ? SECURELINK_OK : SECURELINK_ERR_CRYPTO_FAILED;
}

securelink_err_t securelink_sha384(
    const uint8_t *data, size_t data_len,
    uint8_t out_digest[SECURELINK_DIGEST_SIZE]
) {
    if (!data || !out_digest) {
        return SECURELINK_ERR_INVALID_PARAM;
    }
    const mbedtls_md_info_t *md = mbedtls_md_info_from_type(MBEDTLS_MD_SHA384);
    if (!md) {
        return SECURELINK_ERR_CRYPTO_FAILED;
    }
    int ret = mbedtls_md(md, data, data_len, out_digest);
    return (ret == 0) ? SECURELINK_OK : SECURELINK_ERR_CRYPTO_FAILED;
}

bool securelink_constant_time_equal(
    const uint8_t *a,
    const uint8_t *b,
    size_t len
) {
    if (!a || !b) return false;
    uint8_t diff = 0;
    for (size_t i = 0; i < len; ++i) {
        diff |= (a[i] ^ b[i]);
    }
    return (diff == 0);
}

securelink_err_t securelink_aes_gcm_encrypt(
    const uint8_t key[SECURELINK_KEY_SIZE],
    const uint8_t iv[SECURELINK_GCM_NONCE_SIZE],
    const uint8_t *aad, size_t aad_len,
    const uint8_t *plaintext, size_t plaintext_len,
    uint8_t *ciphertext_out,
    uint8_t tag_out[SECURELINK_TAG_SIZE]
) {
    if (!key || !iv || !plaintext || !ciphertext_out || !tag_out) {
        return SECURELINK_ERR_INVALID_PARAM;
    }
    mbedtls_gcm_context gcm;
    mbedtls_gcm_init(&gcm);

    int ret = mbedtls_gcm_setkey(&gcm, MBEDTLS_CIPHER_ID_AES, key, SECURELINK_KEY_SIZE * 8);
    if (ret != 0) {
        mbedtls_gcm_free(&gcm);
        return SECURELINK_ERR_CRYPTO_FAILED;
    }

    ret = mbedtls_gcm_crypt_and_tag(&gcm, MBEDTLS_GCM_ENCRYPT, plaintext_len,
                                    iv, SECURELINK_GCM_NONCE_SIZE,
                                    aad, aad_len,
                                    plaintext, ciphertext_out,
                                    SECURELINK_TAG_SIZE, tag_out);
    mbedtls_gcm_free(&gcm);
    return (ret == 0) ? SECURELINK_OK : SECURELINK_ERR_CRYPTO_FAILED;
}

securelink_err_t securelink_aes_gcm_decrypt(
    const uint8_t key[SECURELINK_KEY_SIZE],
    const uint8_t iv[SECURELINK_GCM_NONCE_SIZE],
    const uint8_t *aad, size_t aad_len,
    const uint8_t *ciphertext, size_t ciphertext_len,
    const uint8_t tag[SECURELINK_TAG_SIZE],
    uint8_t *plaintext_out
) {
    if (!key || !iv || !ciphertext || !tag || !plaintext_out) {
        return SECURELINK_ERR_INVALID_PARAM;
    }
    mbedtls_gcm_context gcm;
    mbedtls_gcm_init(&gcm);

    int ret = mbedtls_gcm_setkey(&gcm, MBEDTLS_CIPHER_ID_AES, key, SECURELINK_KEY_SIZE * 8);
    if (ret != 0) {
        mbedtls_gcm_free(&gcm);
        return SECURELINK_ERR_CRYPTO_FAILED;
    }

    ret = mbedtls_gcm_auth_decrypt(&gcm, ciphertext_len,
                                   iv, SECURELINK_GCM_NONCE_SIZE,
                                   aad, aad_len,
                                   tag, SECURELINK_TAG_SIZE,
                                   ciphertext, plaintext_out);
    mbedtls_gcm_free(&gcm);
    return (ret == 0) ? SECURELINK_OK : SECURELINK_ERR_CRYPTO_FAILED;
}
