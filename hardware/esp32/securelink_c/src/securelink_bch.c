#include "securelink_bch.h"
#include <string.h>

#define GF_POLY 0x211 /* x^9 + x^4 + 1 */
#define BCH_GEN_POLY_LOWER 0x0c5bf84377ebc7ULL /* 54-bit remainder of g(x) */

void securelink_bch_init(securelink_bch_ctx_t *ctx) {
    if (!ctx) return;
    uint32_t val = 1;
    for (int i = 0; i < SECURELINK_BCH_N; ++i) {
        ctx->log2a[i] = (uint16_t)val;
        ctx->a2log[val] = (uint16_t)i;
        val <<= 1;
        if (val & (1 << SECURELINK_BCH_M)) {
            val ^= GF_POLY;
        }
    }
    ctx->log2a[SECURELINK_BCH_N] = 1;
    ctx->a2log[0] = 0;
    ctx->initialized = true;
}

static inline uint16_t gf_mul(const securelink_bch_ctx_t *ctx, uint16_t a, uint16_t b) {
    if (a == 0 || b == 0) return 0;
    return ctx->log2a[(ctx->a2log[a] + ctx->a2log[b]) % SECURELINK_BCH_N];
}

static inline uint16_t gf_div(const securelink_bch_ctx_t *ctx, uint16_t a, uint16_t b) {
    if (a == 0) return 0;
    if (b == 0) return 0;
    return ctx->log2a[(ctx->a2log[a] + SECURELINK_BCH_N - ctx->a2log[b]) % SECURELINK_BCH_N];
}

void securelink_bch_encode(const securelink_bch_ctx_t *ctx,
                           const uint8_t *data, size_t len,
                           uint8_t ecc_out[SECURELINK_BCH_ECC_BYTES]) {
    (void)ctx;
    if (!data || !ecc_out) return;
    uint64_t reg = 0;
    for (size_t i = 0; i < len; ++i) {
        uint8_t byte = data[i];
        for (int b = 7; b >= 0; --b) {
            uint8_t bit = (byte >> b) & 1;
            uint8_t feedback = ((reg >> 53) & 1) ^ bit;
            reg = (reg << 1) & ((1ULL << 54) - 1);
            if (feedback) {
                reg ^= BCH_GEN_POLY_LOWER;
            }
        }
    }
    for (int i = 0; i < SECURELINK_BCH_ECC_BYTES; ++i) {
        int shift = 54 - (i + 1) * 8;
        if (shift >= 0) {
            ecc_out[i] = (uint8_t)((reg >> shift) & 0xff);
        } else {
            ecc_out[i] = (uint8_t)((reg << (-shift)) & 0xff);
        }
    }
}

int securelink_bch_decode(const securelink_bch_ctx_t *ctx,
                          uint8_t *data, size_t len,
                          const uint8_t recv_ecc[SECURELINK_BCH_ECC_BYTES]) {
    if (!ctx || !ctx->initialized || !data || !recv_ecc) return SECURELINK_ERR_INVALID_PARAM;
    if (len == 0 || len > SECURELINK_BCH_MAX_DATA_BYTES) return SECURELINK_ERR_INVALID_PARAM;

    /* 1. Compute syndromes S_1 through S_12 */
    uint16_t syn[2 * SECURELINK_BCH_T];
    bool has_error = false;
    size_t total_bits = len * 8;

    for (int r = 1; r <= 2 * SECURELINK_BCH_T; ++r) {
        uint16_t s_r = 0;
        /* Data bits contribution */
        for (size_t byte_i = 0; byte_i < len; ++byte_i) {
            uint8_t byte = data[byte_i];
            for (int bit_i = 0; bit_i < 8; ++bit_i) {
                if ((byte >> (7 - bit_i)) & 1) {
                    size_t bit_idx = byte_i * 8 + bit_i;
                    size_t p = (SECURELINK_BCH_ECC_BITS + total_bits - 1) - bit_idx;
                    size_t power = (r * p) % SECURELINK_BCH_N;
                    s_r ^= ctx->log2a[power];
                }
            }
        }
        /* ECC bits contribution */
        for (int ecc_byte_i = 0; ecc_byte_i < SECURELINK_BCH_ECC_BYTES; ++ecc_byte_i) {
            uint8_t ecc_byte = recv_ecc[ecc_byte_i];
            for (int bit_i = 0; bit_i < 8; ++bit_i) {
                size_t ecc_bit_idx = ecc_byte_i * 8 + bit_i;
                if (ecc_bit_idx < SECURELINK_BCH_ECC_BITS) {
                    if ((ecc_byte >> (7 - bit_i)) & 1) {
                        size_t p = (SECURELINK_BCH_ECC_BITS - 1) - ecc_bit_idx;
                        size_t power = (r * p) % SECURELINK_BCH_N;
                        s_r ^= ctx->log2a[power];
                    }
                }
            }
        }
        syn[r - 1] = s_r;
        if (s_r != 0) {
            has_error = true;
        }
    }

    if (!has_error) {
        return 0; /* Zero errors */
    }

    /* 2. Berlekamp-Massey algorithm to find error-locator polynomial Lambda(x) */
    uint16_t C[SECURELINK_BCH_T + 2] = {1};
    uint16_t B[SECURELINK_BCH_T + 2] = {1};
    int L = 0;
    int m_step = 1;
    uint16_t b_val = 1;

    for (int r = 1; r <= 2 * SECURELINK_BCH_T; ++r) {
        uint16_t delta = syn[r - 1];
        for (int i = 1; i <= L; ++i) {
            delta ^= gf_mul(ctx, C[i], syn[r - 1 - i]);
        }
        if (delta == 0) {
            m_step++;
        } else {
            uint16_t T[SECURELINK_BCH_T + 2];
            memcpy(T, C, sizeof(T));
            uint16_t scale = gf_div(ctx, delta, b_val);
            for (int i = 0; i <= SECURELINK_BCH_T + 1 - m_step; ++i) {
                if (B[i] != 0) {
                    C[i + m_step] ^= gf_mul(ctx, scale, B[i]);
                }
            }
            if (2 * L <= r - 1) {
                L = r - L;
                memcpy(B, T, sizeof(B));
                b_val = delta;
                m_step = 1;
            } else {
                m_step++;
            }
        }
    }

    if (L > SECURELINK_BCH_T) {
        return SECURELINK_ERR_RECONCILE_FAILED; /* Too many errors */
    }

    /* 3. Chien search to find root locations and flip error bits in data */
    int corrected = 0;
    for (size_t byte_i = 0; byte_i < len; ++byte_i) {
        for (int bit_i = 0; bit_i < 8; ++bit_i) {
            size_t bit_idx = byte_i * 8 + bit_i;
            size_t p = (SECURELINK_BCH_ECC_BITS + total_bits - 1) - bit_idx;
            uint16_t x = ctx->log2a[(SECURELINK_BCH_N - (p % SECURELINK_BCH_N)) % SECURELINK_BCH_N];
            uint16_t val = 0;
            uint16_t x_pow = 1;
            for (int i = 0; i <= L; ++i) {
                val ^= gf_mul(ctx, C[i], x_pow);
                x_pow = gf_mul(ctx, x_pow, x);
            }
            if (val == 0) {
                data[byte_i] ^= (1 << (7 - bit_i));
                corrected++;
            }
        }
    }

    return corrected;
}
