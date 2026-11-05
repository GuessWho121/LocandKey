#include "securelink_feature.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static int compare_floats(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

securelink_err_t securelink_extract_channel_feature(
    const int8_t *csi_raw,
    size_t num_frames,
    size_t subcarriers,
    float *relative_variance,
    float *normalized_feature
) {
    if (!csi_raw || !relative_variance || !normalized_feature) {
        return SECURELINK_ERR_INVALID_PARAM;
    }
    if (num_frames < 4 || subcarriers == 0 || subcarriers > SECURELINK_MAX_FEATURES) {
        return SECURELINK_ERR_INVALID_PARAM;
    }

    /* 1. Calculate subcarrier magnitudes and means */
    static float mean_mag[SECURELINK_MAX_FEATURES];
    memset(mean_mag, 0, sizeof(float) * subcarriers);

    float total_power = 0.0f;
    for (size_t f = 0; f < num_frames; ++f) {
        for (size_t k = 0; k < subcarriers; ++k) {
            size_t idx = (f * subcarriers + k) * 2;
            int8_t real = csi_raw[idx];
            int8_t imag = csi_raw[idx + 1];
            float mag = sqrtf((float)(real * real + imag * imag));
            mean_mag[k] += mag;
            total_power += mag * mag;
        }
    }
    for (size_t k = 0; k < subcarriers; ++k) {
        mean_mag[k] /= (float)num_frames;
    }
    total_power /= (float)(num_frames * subcarriers);

    /* 2. Calculate dynamic temporal variance */
    float temp_var = 0.0f;
    for (size_t f = 0; f < num_frames; ++f) {
        for (size_t k = 0; k < subcarriers; ++k) {
            size_t idx = (f * subcarriers + k) * 2;
            int8_t real = csi_raw[idx];
            int8_t imag = csi_raw[idx + 1];
            float mag = sqrtf((float)(real * real + imag * imag));
            float diff = mag - mean_mag[k];
            temp_var += diff * diff;
        }
    }
    temp_var /= (float)(num_frames * subcarriers);

    float rel_var = temp_var / (total_power > 1e-7f ? total_power : 1e-7f);
    *relative_variance = rel_var;

    /* 3. Median centering and standard deviation normalization */
    static float sorted[SECURELINK_MAX_FEATURES];
    memcpy(sorted, mean_mag, sizeof(float) * subcarriers);
    qsort(sorted, subcarriers, sizeof(float), compare_floats);

    float median_val;
    if (subcarriers % 2 == 1) {
        median_val = sorted[subcarriers / 2];
    } else {
        median_val = 0.5f * (sorted[subcarriers / 2 - 1] + sorted[subcarriers / 2]);
    }

    float var_feat = 0.0f;
    for (size_t k = 0; k < subcarriers; ++k) {
        float centered = mean_mag[k] - median_val;
        var_feat += centered * centered;
    }
    float std_feat = sqrtf(var_feat / (float)subcarriers);
    if (std_feat < 1e-7f) {
        std_feat = 1e-7f;
    }

    for (size_t k = 0; k < subcarriers; ++k) {
        normalized_feature[k] = (mean_mag[k] - median_val) / std_feat;
    }

    return SECURELINK_OK;
}

securelink_err_t securelink_quantize_bits(
    const float *normalized_feature,
    size_t subcarriers,
    float guard_ratio,
    uint8_t *mask_out,
    uint16_t *mask_bytes_out,
    uint8_t *bits_out,
    uint16_t *bits_count_out
) {
    if (!normalized_feature || !mask_out || !mask_bytes_out || !bits_out || !bits_count_out) {
        return SECURELINK_ERR_INVALID_PARAM;
    }

    uint16_t mask_bytes = (uint16_t)((subcarriers + 7) / 8);
    *mask_bytes_out = mask_bytes;
    memset(mask_out, 0, mask_bytes);

    /* 1. Identify indices where |feature| >= guard_ratio */
    static uint16_t raw_reliable[SECURELINK_MAX_FEATURES];
    uint16_t raw_count = 0;

    for (size_t k = 0; k < subcarriers; ++k) {
        if (fabsf(normalized_feature[k]) >= guard_ratio) {
            raw_reliable[raw_count++] = (uint16_t)k;
        }
    }

    /* 2. Truncate to multiple of 8, capped at BCH max data bits */
    uint16_t usable_bits = (raw_count / 8) * 8;
    if (usable_bits > SECURELINK_BCH_MAX_DATA_BITS) {
        usable_bits = SECURELINK_BCH_MAX_DATA_BITS;
    }

    if (usable_bits < SECURELINK_MIN_BITS_DEF) {
        *bits_count_out = usable_bits;
        return SECURELINK_ERR_MIN_BITS;
    }

    /* 3. Populate mask and pack extracted bits */
    size_t out_bytes = usable_bits / 8;
    memset(bits_out, 0, out_bytes);
    uint16_t ones_count = 0;

    for (uint16_t i = 0; i < usable_bits; ++i) {
        uint16_t k = raw_reliable[i];
        /* Set bit in mask_out */
        mask_out[k / 8] |= (1 << (7 - (k % 8)));

        /* Bit is 1 if feature >= 0 */
        if (normalized_feature[k] >= 0.0f) {
            bits_out[i / 8] |= (1 << (7 - (i % 8)));
            ones_count++;
        }
    }

    float balance = (float)ones_count / (float)usable_bits;
    *bits_count_out = usable_bits;

    if (balance < 0.25f || balance > 0.75f) {
        return SECURELINK_ERR_BALANCE;
    }

    return SECURELINK_OK;
}
