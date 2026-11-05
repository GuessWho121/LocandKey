#ifndef SECURELINK_BCH_H
#define SECURELINK_BCH_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "securelink_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint16_t a2log[SECURELINK_BCH_N + 1];
    uint16_t log2a[SECURELINK_BCH_N + 1];
    bool initialized;
} securelink_bch_ctx_t;

/* Initialize GF(2^9) log/antilog tables */
void securelink_bch_init(securelink_bch_ctx_t *ctx);

/* Encode data bytes and output 7 bytes (54 bits) of BCH ECC helper data */
void securelink_bch_encode(const securelink_bch_ctx_t *ctx,
                           const uint8_t *data, size_t len,
                           uint8_t ecc_out[SECURELINK_BCH_ECC_BYTES]);

/* Correct up to t=6 bit errors in data using recv_ecc; returns number of corrected bits or negative error */
int securelink_bch_decode(const securelink_bch_ctx_t *ctx,
                          uint8_t *data, size_t len,
                          const uint8_t recv_ecc[SECURELINK_BCH_ECC_BYTES]);

#ifdef __cplusplus
}
#endif

#endif /* SECURELINK_BCH_H */
