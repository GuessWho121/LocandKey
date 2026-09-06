"""
Resonance - Model Architecture v2
===================================
ConvNeXt U-Net with:
  - Attention Gates in decoder (focus on relevant channel features)
  - CBAM channel attention in bottleneck (what to amplify)
  - Multi-scale loss (accuracy at every resolution)
  - Spectral loss (frequency domain accuracy)
  - Physics-informed spatial correlation penalty

Input  : (batch, 128, 256, 2)  — 128 antennas × 256 subcarriers × I/Q
Output : (batch, 128, 256, 2)  — denoised channel estimate
"""

import tensorflow as tf
from tensorflow.keras import layers, models

# ══════════════════════════════════════════════════════
#  1. BUILDING BLOCKS
# ══════════════════════════════════════════════════════

def convnext_block(x, dim):
    """
    ConvNeXt block: large-kernel depthwise conv + inverted bottleneck.
    Captures long-range spatial correlations across antennas/subcarriers.
    """
    shortcut = x
    x = layers.DepthwiseConv2D(kernel_size=7, padding='same', use_bias=False)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Conv2D(dim * 4, kernel_size=1)(x)
    x = layers.Activation('gelu')(x)
    x = layers.Conv2D(dim, kernel_size=1)(x)
    # ← ADD THIS: project shortcut if channel count differs
    if shortcut.shape[-1] != dim:
        shortcut = layers.Conv2D(dim, kernel_size=1, use_bias=False)(shortcut)
    x = layers.Add()([shortcut, x])
    return x


def cbam_channel_attention(x, reduction=4):
    """
    CBAM Channel Attention: learns WHICH feature maps matter most.
    Helps the bottleneck focus on dominant propagation modes.
    """
    channels = x.shape[-1]
    # Global average + max pooling
    avg = layers.GlobalAveragePooling2D(keepdims=True)(x)
    mx  = layers.GlobalMaxPooling2D(keepdims=True)(x)

    # Shared MLP
    dense1 = layers.Dense(max(channels // reduction, 1), activation='relu')
    dense2 = layers.Dense(channels)

    avg = dense2(dense1(avg))
    mx  = dense2(dense1(mx))

    scale = layers.Activation('sigmoid')(layers.Add()([avg, mx]))
    return layers.Multiply()([x, scale])


def attention_gate(x, g, inter_channels):
    """
    Attention Gate: lets the decoder selectively focus on encoder features.
    g = gating signal from decoder (coarser, deeper)
    x = encoder skip connection (finer, shallower)

    Physically: emphasises antenna/subcarrier regions with strong signal paths
    while suppressing noise-dominated regions.
    """
    # Match spatial dims with 1×1 convs
    theta_x = layers.Conv2D(inter_channels, kernel_size=1, use_bias=False)(x)
    phi_g   = layers.Conv2D(inter_channels, kernel_size=1, use_bias=False)(g)

    # If spatial dims differ, upsample g to match x
    if theta_x.shape[1] != phi_g.shape[1] or theta_x.shape[2] != phi_g.shape[2]:
        phi_g = layers.UpSampling2D(size=(
            theta_x.shape[1] // phi_g.shape[1],
            theta_x.shape[2] // phi_g.shape[2]
        ))(phi_g)

    add   = layers.Activation('relu')(layers.Add()([theta_x, phi_g]))
    psi   = layers.Conv2D(1, kernel_size=1, activation='sigmoid')(add)
    return layers.Multiply()([x, psi])


# ══════════════════════════════════════════════════════
#  2. LOSS FUNCTION
# ══════════════════════════════════════════════════════

def resonance_loss(lambda_phys=0.10, lambda_spec=0.10, lambda_mag=0.05):
    """
    Multi-component loss:

      L = NMSE                          (reconstruction accuracy)
        + lambda_phys × physics_penalty (spatial antenna correlation)
        + lambda_spec × spectral_loss   (frequency domain accuracy)
        + lambda_mag  × magnitude_loss  (channel magnitude accuracy)

    Args:
        lambda_phys : weight for physics (antenna correlation) term
        lambda_spec : weight for spectral (FFT domain) term
        lambda_mag  : weight for magnitude term
    """
    def loss(y_true, y_pred):

        # ── Component 1: NMSE (core accuracy) ──────────────────
        mse          = tf.reduce_mean(tf.square(y_true - y_pred))
        signal_power = tf.reduce_mean(tf.square(y_true))
        nmse         = mse / (signal_power + 1e-8)

        # ── Component 2: Physics — antenna spatial correlation ──
        # Adjacent antennas should have correlated channels (physics of UPA arrays)
        # Axis 1 = antenna dimension
        true_diff    = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        pred_diff    = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        physics_loss = tf.reduce_mean(tf.square(true_diff - pred_diff))

        # ── Component 3: Spectral loss (FFT domain) ─────────────
        # Convert I/Q back to complex: shape (batch, antennas, subcarriers)
        true_complex = tf.cast(y_true[..., 0], tf.complex64) + \
                       1j * tf.cast(y_true[..., 1], tf.complex64)
        pred_complex = tf.cast(y_pred[..., 0], tf.complex64) + \
                       1j * tf.cast(y_pred[..., 1], tf.complex64)

        # FFT along subcarrier axis → time domain impulse response
        true_fft     = tf.signal.fft(true_complex)
        pred_fft     = tf.signal.fft(pred_complex)

        # L2 error in FFT domain (both real and imag parts)
        fft_error    = tf.abs(true_fft - pred_fft)
        spec_loss    = tf.reduce_mean(tf.square(fft_error))
        spec_norm    = tf.reduce_mean(tf.square(tf.abs(true_fft))) + 1e-8
        spec_loss    = spec_loss / spec_norm

        # ── Component 4: Magnitude loss ──────────────────────────
        # |H_true| vs |H_pred| — ensures amplitude accuracy
        true_mag     = tf.sqrt(tf.square(y_true[..., 0]) +
                               tf.square(y_true[..., 1]) + 1e-8)
        pred_mag     = tf.sqrt(tf.square(y_pred[..., 0]) +
                               tf.square(y_pred[..., 1]) + 1e-8)
        mag_loss     = tf.reduce_mean(tf.square(true_mag - pred_mag)) / \
                       (tf.reduce_mean(tf.square(true_mag)) + 1e-8)

        total = (nmse
                 + lambda_phys * physics_loss
                 + lambda_spec * spec_loss
                 + lambda_mag  * mag_loss)

        return total

    return loss


# ══════════════════════════════════════════════════════
#  3. MODEL BUILDER
# ══════════════════════════════════════════════════════

def build_resonance_model(input_shape=(128, 256, 2)):
    """
    ConvNeXt U-Net with Attention Gates and CBAM.
    Encoder: progressively extracts multi-scale channel features
    Bottleneck: CBAM attention highlights dominant propagation modes
    Decoder: attention-gated skip connections for precise reconstruction
    Args:
        input_shape : (antennas, subcarriers, 2)
    Returns:
        tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape)

    # ── ENCODER ────────────────────────────────────────
    x    = layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(inputs)
    x    = layers.LayerNormalization(epsilon=1e-6)(x)
    enc1 = convnext_block(x, 32)                          # (64, 128, 32)

    x    = layers.Conv2D(64, kernel_size=2, strides=2, padding='same')(enc1)
    x    = layers.LayerNormalization(epsilon=1e-6)(x)
    enc2 = convnext_block(x, 64)                          # (32, 64, 64)

    x    = layers.Conv2D(128, kernel_size=2, strides=2, padding='same')(enc2)
    x    = layers.LayerNormalization(epsilon=1e-6)(x)
    enc3 = convnext_block(x, 128)                         # (16, 32, 128)

    # ── BOTTLENECK ─────────────────────────────────────
    x = layers.Conv2D(256, kernel_size=2, strides=2, padding='same')(enc3)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = convnext_block(x, 256)
    x = convnext_block(x, 256)
    x = cbam_channel_attention(x, reduction=4)            # (8, 16, 256)

    # ── DECODER ────────────────────────────────────────
    # Stage 1
    x        = layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(x)
    x        = layers.LayerNormalization(epsilon=1e-6)(x)
    enc3_att = attention_gate(enc3, x, inter_channels=64)
    x        = layers.Concatenate()([x, enc3_att])
    x        = convnext_block(x, 128)                     # (16, 32, 128)

    # Stage 2
    x        = layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(x)
    x        = layers.LayerNormalization(epsilon=1e-6)(x)
    enc2_att = attention_gate(enc2, x, inter_channels=32)
    x        = layers.Concatenate()([x, enc2_att])
    x        = convnext_block(x, 64)                      # (32, 64, 64)

    # Stage 3
    x        = layers.Conv2DTranspose(32, kernel_size=2, strides=2, padding='same')(x)
    x        = layers.LayerNormalization(epsilon=1e-6)(x)
    enc1_att = attention_gate(enc1, x, inter_channels=16)
    x        = layers.Concatenate()([x, enc1_att])
    x        = convnext_block(x, 32)                      # (64, 128, 32)

    # Final upsample
    x        = layers.Conv2DTranspose(16, kernel_size=4, strides=2, padding='same')(x)
    x        = layers.LayerNormalization(epsilon=1e-6)(x)
    x        = layers.Activation('gelu')(x)

    outputs  = layers.Conv2D(2, kernel_size=1, activation='linear',
                            dtype='float32')(x)           # (128, 256, 2)

    model = models.Model(inputs, outputs, name="Resonance_v2_ConvNeXt_Attention")
    return model


# ══════════════════════════════════════════════════════
#  4. NMSE METRIC (logged separately from loss)
# ══════════════════════════════════════════════════════

def nmse_metric(y_true, y_pred):
    """Standalone NMSE for monitoring — not used in backprop."""
    mse    = tf.reduce_mean(tf.square(y_true - y_pred))
    energy = tf.reduce_mean(tf.square(y_true))
    return mse / (energy + 1e-8)


def nmse_db_metric(y_true, y_pred):
    """NMSE in dB — the standard telecom reporting format."""
    nmse = nmse_metric(y_true, y_pred)
    # Clip to avoid log(0)
    nmse = tf.maximum(nmse, 1e-10)
    return 10.0 * tf.experimental.numpy.log10(nmse)


# ══════════════════════════════════════════════════════
#  SANITY CHECK
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    model = build_resonance_model(input_shape=(128, 256, 2))
    model.summary()
    print(f"\nTotal params: {model.count_params():,}")

    # Verify output shape
    import numpy as np
    dummy = np.random.randn(2, 128, 256, 2).astype(np.float32)
    out   = model(dummy, training=False)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    assert out.shape == dummy.shape, "Shape mismatch!"
    print("✅ Shape check passed.")