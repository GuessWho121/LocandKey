#ifndef SECURELINK_PROTOCOL_H
#define SECURELINK_PROTOCOL_H

#include "securelink_types.h"
#include "securelink_bch.h"
#include "securelink_feature.h"
#include "securelink_crypto.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Replay prevention cache */
typedef struct {
    uint8_t nonces[SECURELINK_NONCE_CACHE_SIZE][SECURELINK_NONCE_SIZE];
    uint8_t count;
    uint8_t head;
} securelink_replay_cache_t;

void securelink_replay_cache_init(securelink_replay_cache_t *cache);
bool securelink_replay_cache_check_and_add(securelink_replay_cache_t *cache, const uint8_t nonce[SECURELINK_NONCE_SIZE]);

/* Alice: Process local CSI, quantize, encode BCH helper, and build offer packet */
securelink_err_t securelink_alice_create_offer(
    const int8_t *alice_csi_raw,
    size_t num_frames,
    size_t subcarriers,
    const uint8_t auth_secret[SECURELINK_KEY_SIZE],
    const uint8_t nonce[SECURELINK_NONCE_SIZE],
    const uint8_t *transcript, size_t transcript_len,
    const securelink_bch_ctx_t *bch_ctx,
    securelink_offer_t *offer_out,
    securelink_alice_session_t *session_out
);

/* Bob: Authenticate offer, check replay, reconcile bits, and generate confirmation reply */
securelink_err_t securelink_bob_accept_offer(
    const int8_t *bob_csi_raw,
    size_t num_frames,
    size_t subcarriers,
    const securelink_offer_t *offer,
    const uint8_t auth_secret[SECURELINK_KEY_SIZE],
    const uint8_t expected_nonce[SECURELINK_NONCE_SIZE],
    const uint8_t *transcript, size_t transcript_len,
    const securelink_bch_ctx_t *bch_ctx,
    securelink_replay_cache_t *replay_cache,
    securelink_reply_t *reply_out,
    securelink_bob_session_t *session_out
);

/* Alice: Verify Bob's confirmation tag before activating session key */
securelink_err_t securelink_alice_finalize(
    securelink_alice_session_t *session,
    const securelink_reply_t *reply
);

/* Derive authenticated pre-shared secret fallback key */
securelink_err_t securelink_derive_fallback_key(
    const uint8_t auth_secret[SECURELINK_KEY_SIZE],
    const uint8_t nonce[SECURELINK_NONCE_SIZE],
    const uint8_t *transcript, size_t transcript_len,
    uint8_t key_out[SECURELINK_KEY_SIZE]
);

#ifdef __cplusplus
}
#endif

#endif /* SECURELINK_PROTOCOL_H */
