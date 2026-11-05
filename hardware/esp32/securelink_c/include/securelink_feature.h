#ifndef SECURELINK_FEATURE_H
#define SECURELINK_FEATURE_H

#include "securelink_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Extract channel metrics from raw CSI packet frames.
 * csi_raw: interleaved I/Q signed byte pairs [num_frames * subcarriers * 2]
 * num_frames: number of consecutive CSI packet frames (minimum 4)
 * subcarriers: number of OFDM subcarriers (e.g. 64)
 * relative_variance: output relative temporal variance
 * normalized_feature: output normalized subcarrier feature array [subcarriers]
 */
securelink_err_t securelink_extract_channel_feature(
    const int8_t *csi_raw,
    size_t num_frames,
    size_t subcarriers,
    float *relative_variance,
    float *normalized_feature
);

/*
 * Quantize normalized feature into reliable bits using guard-band thresholding.
 * normalized_feature: input feature array [subcarriers]
 * subcarriers: number of subcarriers
 * guard_ratio: threshold exclusion factor (e.g. 0.25f or 0.15f)
 * mask_out: output reliability mask bit array
 * mask_bytes_out: output size of mask in bytes
 * bits_out: packed binary bits of reliable features
 * bits_count_out: count of extracted bits (truncated to whole bytes)
 */
securelink_err_t securelink_quantize_bits(
    const float *normalized_feature,
    size_t subcarriers,
    float guard_ratio,
    uint8_t *mask_out,
    uint16_t *mask_bytes_out,
    uint8_t *bits_out,
    uint16_t *bits_count_out
);

#ifdef __cplusplus
}
#endif

#endif /* SECURELINK_FEATURE_H */
