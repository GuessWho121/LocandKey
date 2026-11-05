"""Native Wi-Fi IoT CSI Residual Denoiser (1D Frequency ConvNeXt U-Net).

Tensors use (batch, 2, subcarriers), where:
- channel 0: Real component
- channel 1: Imaginary component
- subcarriers: 64 for 802.11n 20 MHz HT20
"""

import torch
from torch import nn


def nmse_metric(y_true, y_pred):
    y_true, y_pred = y_true.float(), y_pred.float()
    return (y_true - y_pred).square().mean() / (y_true.square().mean() + 1e-8)


def nmse_db_metric(y_true, y_pred):
    return 10.0 * nmse_metric(y_true, y_pred).clamp_min(1e-10).log10()


def resonance_loss(lambda_mag=0.05, lambda_nmse=0.0):
    def loss(y_true, y_pred, noisy):
        y_true, y_pred, noisy = y_true.float(), y_pred.float(), noisy.float()
        correction_error = (y_pred - y_true).square().flatten(1).sum(1)
        noise_energy = (noisy - y_true).square().flatten(1).sum(1)
        true_mag = (y_true.square().sum(dim=1) + 1e-8).sqrt()
        pred_mag = (y_pred.square().sum(dim=1) + 1e-8).sqrt()
        mag_loss = (true_mag - pred_mag).square().mean() / (true_mag.square().mean() + 1e-8)
        return ((correction_error / noise_energy.clamp_min(1e-8)).mean()
                + lambda_nmse * nmse_metric(y_true, y_pred) + lambda_mag * mag_loss)
    return loss


class ChannelNorm1d(nn.LayerNorm):
    """LayerNormalization over channels independently per subcarrier."""

    def __init__(self, channels: int):
        super().__init__(channels, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, L) -> (N, L, C) -> LayerNorm -> (N, C, L)
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)


class ConvNeXtBlock1d(nn.Module):
    """1D ConvNeXt block operating across subcarrier frequency axis."""

    def __init__(self, channels: int, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=3, groups=channels, bias=False),
            ChannelNorm1d(channels),
            nn.Conv1d(channels, dim * 4, 1),
            nn.GELU(),
            nn.Conv1d(dim * 4, dim, 1),
        )
        self.shortcut = nn.Identity() if channels == dim else nn.Conv1d(channels, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class ChannelAttention1d(nn.Module):
    """1D Squeeze-and-Excitation / Channel Attention over frequency."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, max(channels // reduction, 1), 1),
            nn.ReLU(),
            nn.Conv1d(max(channels // reduction, 1), channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.mlp(x.mean(2, keepdim=True)) + self.mlp(x.amax(2, keepdim=True))
        return x * scale.sigmoid()


class AttentionGate1d(nn.Module):
    """Attention Gate for 1D skip connections."""

    def __init__(self, channels: int, inter_channels: int):
        super().__init__()
        self.theta = nn.Conv1d(channels, inter_channels, 1, bias=False)
        self.phi = nn.Conv1d(channels, inter_channels, 1, bias=False)
        self.psi = nn.Conv1d(inter_channels, 1, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return x * self.psi(torch.relu(self.theta(x) + self.phi(g))).sigmoid()


class WiFiResonanceModel(nn.Module):
    """Native residual frequency denoiser for 1x1 SISO Wi-Fi CSI (e.g. 64 subcarriers)."""

    def __init__(self, subcarriers: int = 64, residual: bool = True, safety_factor: float = 2.0):
        super().__init__()
        if subcarriers <= 0 or subcarriers % 16 != 0:
            raise ValueError(f"subcarriers must be a positive multiple of 16, got {subcarriers}")
        if safety_factor <= 0:
            raise ValueError("safety_factor must be positive")
        self.subcarriers = subcarriers
        self.residual = residual
        self.safety_factor = safety_factor

        # Encoder: 64 -> 32 -> 16 -> 8
        self.encoder = nn.ModuleList()
        for cin, cout, kernel in ((2, 32, 4), (32, 64, 2), (64, 128, 2)):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(cin, cout, kernel, stride=2, padding=1 if kernel == 4 else 0),
                    ChannelNorm1d(cout),
                    ConvNeXtBlock1d(cout, cout),
                )
            )

        # Bottleneck: 8 subcarriers
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 256, 2, stride=2),
            ChannelNorm1d(256),
            ConvNeXtBlock1d(256, 256),
            ConvNeXtBlock1d(256, 256),
            ChannelAttention1d(256),
        )

        # Decoder: 8 -> 16 -> 32 -> 64
        self.ups = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for cin, cout in ((256, 128), (128, 64), (64, 32)):
            self.ups.append(nn.Sequential(nn.ConvTranspose1d(cin, cout, 2, stride=2), ChannelNorm1d(cout)))
            self.gates.append(AttentionGate1d(cout, cout // 2))
            self.decoder.append(ConvNeXtBlock1d(cout * 2, cout))

        self.final = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1),
            ChannelNorm1d(16),
            nn.GELU(),
        )
        self.output = nn.Conv1d(16, 2, 1)

        if residual:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] != self.subcarriers:
            raise ValueError(f"Expected (N, 2, {self.subcarriers}), got {tuple(x.shape)}")
        noisy = x
        skips = []
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)
        x = self.bottleneck(x)
        for up, gate, decoder, skip in zip(self.ups, self.gates, self.decoder, reversed(skips)):
            x = up(x)
            x = decoder(torch.cat((x, gate(skip, x)), dim=1))
        x = self.final(x)

        with torch.autocast(device_type=x.device.type, enabled=False):
            correction = self.output(x.float())
            if not self.residual:
                return correction

            # If snr_db is provided, scale and clip correction against expected noise power
            if snr_db is not None:
                snr = snr_db.float().reshape(-1, 1, 1)
                expected_noise_std = torch.sqrt(
                    torch.clamp(1.0 / torch.pow(10.0, snr / 10.0), min=1e-10)
                )
                max_norm = self.safety_factor * expected_noise_std
                correction_norm = torch.sqrt(torch.sum(correction ** 2, dim=(1, 2), keepdim=True) + 1e-8)
                clip_coef = torch.clamp(max_norm / correction_norm, max=1.0)
                correction = correction * clip_coef

            return noisy.float() + correction


def build_wifi_resonance_model(subcarriers: int = 64, residual: bool = True, safety_factor: float = 2.0):
    return WiFiResonanceModel(subcarriers=subcarriers, residual=residual, safety_factor=safety_factor)


if __name__ == "__main__":
    model = build_wifi_resonance_model()
    dummy = torch.randn(4, 2, 64)
    snr = torch.tensor([0.0, 10.0, 20.0, 30.0])
    out = model(dummy, snr)
    assert out.shape == dummy.shape and torch.isfinite(out).all()
    # At initialization with residual=True, output equals input exactly
    assert torch.allclose(out, dummy, atol=1e-6)
    loss = resonance_loss()(dummy, out, dummy)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    num_params = sum(p.numel() for p in model.parameters())
    print(f"WiFiResonanceModel forward/backward passed; parameters: {num_params:,}")
