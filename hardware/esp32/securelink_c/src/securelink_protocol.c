#include "securelink_protocol.h"
#include <string.h>

static const char OFFER_PREFIX[] = "securelink/offer/v2";
static const char CHANNEL_INFO_PREFIX[] = "securelink/channel/v4";
static const char CONFIRM_PREFIX[] = "securelink/bob-confirm/v2";
static const char FALLBACK_INFO_PREFIX[] = "securelink/fallback/v4";

void securelink_replay_cache_init(securelink_replay_cache_t *cache) {
    if (!cache) return;
    memset(cache, 0, sizeof(*cache));
}

bool securelink_replay_cache_check_and_add(securelink_replay_cache_t *cache, const uint8_t nonce[SECURELINK_NONCE_SIZE]) {
    if (!cache || !nonce) return false;
    for (uint8_t i = 0; i < cache->count; ++i) {
        if (memcmp(cache->nonces[i], nonce, SECURELINK_NONCE_SIZE) == 0) {
            return true; /* Replay detected */
        }
    }
    /* Add to ring buffer */
    memcpy(cache->nonces[cache->head], nonce, SECURELINK_NONCE_SIZE);
    cache->head = (cache->head + 1) % SECURELINK_NONCE_CACHE_SIZE;
    if (cache->count < SECURELINK_NONCE_CACHE_SIZE) {
        cache->count++;
    }
    return false;
}

static size_t build_offer_payload(
    const securelink_offer_t *offer,
    uint8_t *buf_out,
    size_t buf_max
) {
    size_t prefix_len = sizeof(OFFER_PREFIX) - 1; /* 19 */
    size_t header_len = 14; /* >HIII */
    size_t required = prefix_len + header_len + SECURELINK_NONCE_SIZE +
                      SECURELINK_DIGEST_SIZE + offer->mask_bytes + SECURELINK_BCH_ECC_BYTES;
    if (required > buf_max) return 0;

    size_t offset = 0;
    memcpy(buf_out + offset, OFFER_PREFIX, prefix_len);
    offset += prefix_len;

    /* Big-endian: len(nonce)=16, feature_count, mask_bytes, helper_bytes=7 */
    buf_out[offset++] = (uint8_t)((SECURELINK_NONCE_SIZE >> 8) & 0xff);
    buf_out[offset++] = (uint8_t)(SECURELINK_NONCE_SIZE & 0xff);

    buf_out[offset++] = (uint8_t)((offer->feature_count >> 24) & 0xff);
    buf_out[offset++] = (uint8_t)((offer->feature_count >> 16) & 0xff);
    buf_out[offset++] = (uint8_t)((offer->feature_count >> 8) & 0xff);
    buf_out[offset++] = (uint8_t)(offer->feature_count & 0xff);

    buf_out[offset++] = (uint8_t)((offer->mask_bytes >> 24) & 0xff);
    buf_out[offset++] = (uint8_t)((offer->mask_bytes >> 16) & 0xff);
    buf_out[offset++] = (uint8_t)((offer->mask_bytes >> 8) & 0xff);
    buf_out[offset++] = (uint8_t)(offer->mask_bytes & 0xff);

    buf_out[offset++] = 0;
    buf_out[offset++] = 0;
    buf_out[offset++] = 0;
    buf_out[offset++] = SECURELINK_BCH_ECC_BYTES;

    memcpy(buf_out + offset, offer->nonce, SECURELINK_NONCE_SIZE);
    offset += SECURELINK_NONCE_SIZE;

    memcpy(buf_out + offset, offer->transcript_digest, SECURELINK_DIGEST_SIZE);
    offset += SECURELINK_DIGEST_SIZE;

    memcpy(buf_out + offset, offer->reliability_mask, offer->mask_bytes);
    offset += offer->mask_bytes;

    memcpy(buf_out + offset, offer->helper_data, SECURELINK_BCH_ECC_BYTES);
    offset += SECURELINK_BCH_ECC_BYTES;

    return offset;
}

static void compute_offer_digest(
    const uint8_t *payload, size_t payload_len,
    const uint8_t auth_tag[SECURELINK_DIGEST_SIZE],
    uint8_t digest_out[SECURELINK_DIGEST_SIZE]
) {
    uint8_t full_buf[1024];
    if (payload_len + SECURELINK_DIGEST_SIZE <= sizeof(full_buf)) {
        memcpy(full_buf, payload, payload_len);
        memcpy(full_buf + payload_len, auth_tag, SECURELINK_DIGEST_SIZE);
        securelink_sha384(full_buf, payload_len + SECURELINK_DIGEST_SIZE, digest_out);
    }
}

securelink_err_t securelink_derive_fallback_key(
    const uint8_t auth_secret[SECURELINK_KEY_SIZE],
    const uint8_t nonce[SECURELINK_NONCE_SIZE],
    const uint8_t *transcript, size_t transcript_len,
    uint8_t key_out[SECURELINK_KEY_SIZE]
) {
    uint8_t t_digest[SECURELINK_DIGEST_SIZE];
    securelink_sha384(transcript, transcript_len, t_digest);

    uint8_t info[sizeof(FALLBACK_INFO_PREFIX) - 1 + SECURELINK_DIGEST_SIZE];
    memcpy(info, FALLBACK_INFO_PREFIX, sizeof(FALLBACK_INFO_PREFIX) - 1);
    memcpy(info + sizeof(FALLBACK_INFO_PREFIX) - 1, t_digest, SECURELINK_DIGEST_SIZE);

    return securelink_hkdf_sha384(auth_secret, SECURELINK_KEY_SIZE,
                                  nonce, SECURELINK_NONCE_SIZE,
                                  info, sizeof(info),
                                  key_out, SECURELINK_KEY_SIZE);
}

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
) {
    if (!alice_csi_raw || !auth_secret || !nonce || !bch_ctx || !offer_out || !session_out) {
        return SECURELINK_ERR_INVALID_PARAM;
    }

    memset(offer_out, 0, sizeof(*offer_out));
    memset(session_out, 0, sizeof(*session_out));
    memcpy(offer_out->nonce, nonce, SECURELINK_NONCE_SIZE);
    securelink_sha384(transcript, transcript_len, offer_out->transcript_digest);
    offer_out->feature_count = (uint16_t)subcarriers;

    /* 1. Extract channel features */
    float rel_variance = 0.0f;
    static float normalized_feature[SECURELINK_MAX_FEATURES];
    securelink_err_t err = securelink_extract_channel_feature(
        alice_csi_raw, num_frames, subcarriers, &rel_variance, normalized_feature
    );
    if (err != SECURELINK_OK || rel_variance < SECURELINK_MIN_VARIANCE_DEF) {
        /* Variance failure: fallback */
        securelink_derive_fallback_key(auth_secret, nonce, transcript, transcript_len, session_out->key);
        session_out->status = SECURELINK_STATUS_FALLBACK;
        return SECURELINK_ERR_VARIANCE;
    }

    /* 2. Quantize bits */
    uint8_t bits[SECURELINK_BCH_MAX_DATA_BYTES];
    uint16_t bits_count = 0;
    err = securelink_quantize_bits(
        normalized_feature, subcarriers, SECURELINK_GUARD_RATIO_DEF,
        offer_out->reliability_mask, &offer_out->mask_bytes,
        bits, &bits_count
    );
    if (err != SECURELINK_OK) {
        /* Bit count or balance failure: fallback */
        securelink_derive_fallback_key(auth_secret, nonce, transcript, transcript_len, session_out->key);
        session_out->status = SECURELINK_STATUS_FALLBACK;
        return err;
    }

    /* 3. BCH encode helper data */
    size_t data_bytes = bits_count / 8;
    securelink_bch_encode(bch_ctx, bits, data_bytes, offer_out->helper_data);

    /* 4. Construct payload and compute HMAC auth tag */
    static uint8_t payload[512];
    size_t payload_len = build_offer_payload(offer_out, payload, sizeof(payload));
    securelink_hmac_sha384(auth_secret, SECURELINK_KEY_SIZE, payload, payload_len, offer_out->auth_tag);

    /* 5. Compute offer digest */
    compute_offer_digest(payload, payload_len, offer_out->auth_tag, session_out->offer_digest);

    /* 6. Derive Alice's candidate session key */
    uint8_t ikm[SECURELINK_KEY_SIZE + SECURELINK_BCH_MAX_DATA_BYTES];
    memcpy(ikm, auth_secret, SECURELINK_KEY_SIZE);
    memcpy(ikm + SECURELINK_KEY_SIZE, bits, data_bytes);

    uint8_t info[sizeof(CHANNEL_INFO_PREFIX) - 1 + SECURELINK_DIGEST_SIZE];
    memcpy(info, CHANNEL_INFO_PREFIX, sizeof(CHANNEL_INFO_PREFIX) - 1);
    memcpy(info + sizeof(CHANNEL_INFO_PREFIX) - 1, session_out->offer_digest, SECURELINK_DIGEST_SIZE);

    securelink_hkdf_sha384(ikm, SECURELINK_KEY_SIZE + data_bytes,
                          nonce, SECURELINK_NONCE_SIZE,
                          info, sizeof(info),
                          session_out->key, SECURELINK_KEY_SIZE);

    session_out->status = SECURELINK_STATUS_IDLE; /* Waiting for confirmation */
    return SECURELINK_OK;
}

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
) {
    if (!bob_csi_raw || !offer || !auth_secret || !expected_nonce || !bch_ctx || !reply_out || !session_out) {
        return SECURELINK_ERR_INVALID_PARAM;
    }
    memset(reply_out, 0, sizeof(*reply_out));
    memset(session_out, 0, sizeof(*session_out));

    /* 1. Verify nonce and replay cache */
    if (!securelink_constant_time_equal(offer->nonce, expected_nonce, SECURELINK_NONCE_SIZE)) {
        return SECURELINK_ERR_AUTH_FAILED;
    }
    if (replay_cache && securelink_replay_cache_check_and_add(replay_cache, offer->nonce)) {
        return SECURELINK_ERR_REPLAY;
    }

    /* 2. Verify transcript digest */
    uint8_t expected_t_digest[SECURELINK_DIGEST_SIZE];
    securelink_sha384(transcript, transcript_len, expected_t_digest);
    if (!securelink_constant_time_equal(offer->transcript_digest, expected_t_digest, SECURELINK_DIGEST_SIZE)) {
        return SECURELINK_ERR_AUTH_FAILED;
    }

    /* 3. Verify offer HMAC authentication tag */
    static uint8_t bob_payload[512];
    size_t payload_len = build_offer_payload(offer, bob_payload, sizeof(bob_payload));
    uint8_t expected_tag[SECURELINK_DIGEST_SIZE];
    securelink_hmac_sha384(auth_secret, SECURELINK_KEY_SIZE, bob_payload, payload_len, expected_tag);
    if (!securelink_constant_time_equal(offer->auth_tag, expected_tag, SECURELINK_DIGEST_SIZE)) {
        return SECURELINK_ERR_AUTH_FAILED;
    }

    /* 4. Compute offer digest */
    compute_offer_digest(bob_payload, payload_len, offer->auth_tag, reply_out->offer_digest);

    /* 5. Extract Bob's channel metrics */
    float rel_variance = 0.0f;
    static float bob_normalized_feature[SECURELINK_MAX_FEATURES];
    securelink_err_t err = securelink_extract_channel_feature(
        bob_csi_raw, num_frames, subcarriers, &rel_variance, bob_normalized_feature
    );
    if (err != SECURELINK_OK || rel_variance < SECURELINK_MIN_VARIANCE_DEF) {
        securelink_derive_fallback_key(auth_secret, offer->nonce, transcript, transcript_len, session_out->key);
        session_out->status = SECURELINK_STATUS_FALLBACK;
        return SECURELINK_ERR_VARIANCE;
    }

    /* 6. Sample Bob's bits under Alice's reliability mask */
    uint8_t bob_bits[SECURELINK_BCH_MAX_DATA_BYTES];
    memset(bob_bits, 0, sizeof(bob_bits));
    uint16_t reliable_bits = 0;

    for (size_t k = 0; k < subcarriers && reliable_bits < SECURELINK_BCH_MAX_DATA_BITS; ++k) {
        if ((offer->reliability_mask[k / 8] >> (7 - (k % 8))) & 1) {
            if (bob_normalized_feature[k] >= 0.0f) {
                bob_bits[reliable_bits / 8] |= (1 << (7 - (reliable_bits % 8)));
            }
            reliable_bits++;
        }
    }

    if (reliable_bits < SECURELINK_MIN_BITS_DEF || (reliable_bits % 8) != 0) {
        securelink_derive_fallback_key(auth_secret, offer->nonce, transcript, transcript_len, session_out->key);
        session_out->status = SECURELINK_STATUS_FALLBACK;
        return SECURELINK_ERR_MIN_BITS;
    }

    /* 7. Reconcile Bob's bits using BCH decode */
    size_t data_bytes = reliable_bits / 8;
    int corrected = securelink_bch_decode(bch_ctx, bob_bits, data_bytes, offer->helper_data);
    if (corrected < 0) {
        /* Uncorrectable errors: fallback */
        securelink_derive_fallback_key(auth_secret, offer->nonce, transcript, transcript_len, session_out->key);
        session_out->status = SECURELINK_STATUS_FALLBACK;
        return SECURELINK_ERR_RECONCILE_FAILED;
    }

    session_out->corrected_bits = (uint16_t)corrected;
    session_out->correction_rate = (float)corrected / (float)reliable_bits;

    /* 8. Derive Bob's session key */
    uint8_t ikm[SECURELINK_KEY_SIZE + SECURELINK_BCH_MAX_DATA_BYTES];
    memcpy(ikm, auth_secret, SECURELINK_KEY_SIZE);
    memcpy(ikm + SECURELINK_KEY_SIZE, bob_bits, data_bytes);

    uint8_t info[sizeof(CHANNEL_INFO_PREFIX) - 1 + SECURELINK_DIGEST_SIZE];
    memcpy(info, CHANNEL_INFO_PREFIX, sizeof(CHANNEL_INFO_PREFIX) - 1);
    memcpy(info + sizeof(CHANNEL_INFO_PREFIX) - 1, reply_out->offer_digest, SECURELINK_DIGEST_SIZE);

    securelink_hkdf_sha384(ikm, SECURELINK_KEY_SIZE + data_bytes,
                          offer->nonce, SECURELINK_NONCE_SIZE,
                          info, sizeof(info),
                          session_out->key, SECURELINK_KEY_SIZE);

    /* 9. Compute Bob's confirmation tag */
    uint8_t confirm_data[sizeof(CONFIRM_PREFIX) - 1 + SECURELINK_DIGEST_SIZE];
    memcpy(confirm_data, CONFIRM_PREFIX, sizeof(CONFIRM_PREFIX) - 1);
    memcpy(confirm_data + sizeof(CONFIRM_PREFIX) - 1, reply_out->offer_digest, SECURELINK_DIGEST_SIZE);

    securelink_hmac_sha384(session_out->key, SECURELINK_KEY_SIZE,
                          confirm_data, sizeof(confirm_data),
                          reply_out->confirmation_tag);

    session_out->status = SECURELINK_STATUS_ACTIVE;
    return SECURELINK_OK;
}

securelink_err_t securelink_alice_finalize(
    securelink_alice_session_t *session,
    const securelink_reply_t *reply
) {
    if (!session || !reply) return SECURELINK_ERR_INVALID_PARAM;

    /* 1. Verify offer digest matches */
    if (!securelink_constant_time_equal(session->offer_digest, reply->offer_digest, SECURELINK_DIGEST_SIZE)) {
        return SECURELINK_ERR_CONFIRM_FAILED;
    }

    /* 2. Verify Bob's confirmation tag */
    uint8_t confirm_data[sizeof(CONFIRM_PREFIX) - 1 + SECURELINK_DIGEST_SIZE];
    memcpy(confirm_data, CONFIRM_PREFIX, sizeof(CONFIRM_PREFIX) - 1);
    memcpy(confirm_data + sizeof(CONFIRM_PREFIX) - 1, session->offer_digest, SECURELINK_DIGEST_SIZE);

    uint8_t expected_tag[SECURELINK_DIGEST_SIZE];
    securelink_hmac_sha384(session->key, SECURELINK_KEY_SIZE,
                          confirm_data, sizeof(confirm_data),
                          expected_tag);

    if (!securelink_constant_time_equal(reply->confirmation_tag, expected_tag, SECURELINK_DIGEST_SIZE)) {
        return SECURELINK_ERR_CONFIRM_FAILED;
    }

    session->status = SECURELINK_STATUS_ACTIVE;
    return SECURELINK_OK;
}
