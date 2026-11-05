/**
 * @file securelink_tinyml.h
 * @brief On-chip TinyML 1D depthwise-separable CNN denoiser for ESP32.
 * Zero-dependency C implementation for SISO Wi-Fi CSI (64 subcarriers).
 */

#ifndef SECURELINK_TINYML_H
#define SECURELINK_TINYML_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SECURELINK_TINYML_SUBCARRIERS 64

/**
 * @brief Executes on-device TinyML neural denoiser on raw CSI subcarrier frame.
 * @param in_iq: Input CSI array of shape [2][64] (channel 0: Real/I, channel 1: Imag/Q).
 * @param out_iq: Output denoised CSI array of shape [2][64].
 */
void securelink_tinyml_denoise(const float in_iq[2][SECURELINK_TINYML_SUBCARRIERS],
                              float out_iq[2][SECURELINK_TINYML_SUBCARRIERS]);

#ifdef __cplusplus
}
#endif

#endif /* SECURELINK_TINYML_H */
