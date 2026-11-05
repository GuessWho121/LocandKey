/**
 * @file securelink_tinyml.c
 * @brief Zero-dependency TinyML 1D depthwise-separable CNN denoiser execution engine.
 */

#include "securelink_tinyml.h"
#include "securelink_tinyml_weights.h"

#define SUBCARRIERS SECURELINK_TINYML_SUBCARRIERS

static inline float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

void securelink_tinyml_denoise(const float in_iq[2][SUBCARRIERS],
                              float out_iq[2][SUBCARRIERS]) {
    /* Ping-pong buffers: 32 channels x 64 subcarriers = 8 KB each */
    static float buf_a[32][SUBCARRIERS];
    static float buf_b[32][SUBCARRIERS];

    /* 1. Input projection: Conv1d(2 -> 16, k=1) + ReLU */
    for (int s = 0; s < SUBCARRIERS; s++) {
        for (int c = 0; c < 16; c++) {
            float val = tinyml_in_conv_b[c] +
                        tinyml_in_conv_w[c * 2 + 0] * in_iq[0][s] +
                        tinyml_in_conv_w[c * 2 + 1] * in_iq[1][s];
            buf_a[c][s] = relu(val);
        }
    }

    /* 2. Block 1: Depthwise Conv1d(16 -> 16, k=5, p=2) */
    for (int c = 0; c < 16; c++) {
        for (int s = 0; s < SUBCARRIERS; s++) {
            float sum = tinyml_dw1_b[c];
            for (int k = 0; k < 5; k++) {
                int in_idx = s + k - 2;
                if (in_idx >= 0 && in_idx < SUBCARRIERS) {
                    sum += tinyml_dw1_w[c * 5 + k] * buf_a[c][in_idx];
                }
            }
            buf_b[c][s] = sum;
        }
    }

    /* Pointwise Conv1d(16 -> 32, k=1) + ReLU */
    for (int s = 0; s < SUBCARRIERS; s++) {
        for (int co = 0; co < 32; co++) {
            float sum = tinyml_pw1_b[co];
            for (int ci = 0; ci < 16; ci++) {
                sum += tinyml_pw1_w[co * 16 + ci] * buf_b[ci][s];
            }
            buf_a[co][s] = relu(sum);
        }
    }

    /* 3. Block 2: Depthwise Conv1d(32 -> 32, k=5, p=2) */
    for (int c = 0; c < 32; c++) {
        for (int s = 0; s < SUBCARRIERS; s++) {
            float sum = tinyml_dw2_b[c];
            for (int k = 0; k < 5; k++) {
                int in_idx = s + k - 2;
                if (in_idx >= 0 && in_idx < SUBCARRIERS) {
                    sum += tinyml_dw2_w[c * 5 + k] * buf_a[c][in_idx];
                }
            }
            buf_b[c][s] = sum;
        }
    }

    /* Pointwise Conv1d(32 -> 32, k=1) + ReLU */
    for (int s = 0; s < SUBCARRIERS; s++) {
        for (int co = 0; co < 32; co++) {
            float sum = tinyml_pw2_b[co];
            for (int ci = 0; ci < 32; ci++) {
                sum += tinyml_pw2_w[co * 32 + ci] * buf_b[ci][s];
            }
            buf_a[co][s] = relu(sum);
        }
    }

    /* 4. Block 3: Depthwise Conv1d(32 -> 32, k=5, p=2) */
    for (int c = 0; c < 32; c++) {
        for (int s = 0; s < SUBCARRIERS; s++) {
            float sum = tinyml_dw3_b[c];
            for (int k = 0; k < 5; k++) {
                int in_idx = s + k - 2;
                if (in_idx >= 0 && in_idx < SUBCARRIERS) {
                    sum += tinyml_dw3_w[c * 5 + k] * buf_a[c][in_idx];
                }
            }
            buf_b[c][s] = sum;
        }
    }

    /* Pointwise Conv1d(32 -> 16, k=1) + ReLU */
    for (int s = 0; s < SUBCARRIERS; s++) {
        for (int co = 0; co < 16; co++) {
            float sum = tinyml_pw3_b[co];
            for (int ci = 0; ci < 32; ci++) {
                sum += tinyml_pw3_w[co * 32 + ci] * buf_b[ci][s];
            }
            buf_a[co][s] = relu(sum);
        }
    }

    /* 5. Output projection: Conv1d(16 -> 2, k=1) + Residual Addition */
    for (int s = 0; s < SUBCARRIERS; s++) {
        for (int co = 0; co < 2; co++) {
            float correction = tinyml_out_conv_b[co];
            for (int ci = 0; ci < 16; ci++) {
                correction += tinyml_out_conv_w[co * 16 + ci] * buf_a[ci][s];
            }
            out_iq[co][s] = in_iq[co][s] + correction;
        }
    }
}
