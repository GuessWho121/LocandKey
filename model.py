"""LocandKey ConvNeXt U-Net. Tensors use (batch, 2, antennas, subcarriers)."""

import torch
from torch import nn


class ChannelNorm(nn.LayerNorm):
    """Match Keras LayerNormalization over channels, independently per pixel."""

    def __init__(self, channels):
        super().__init__(channels, eps=1e-6)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False),
            ChannelNorm(channels), nn.Conv2d(channels, dim * 4, 1),
            nn.GELU(), nn.Conv2d(dim * 4, dim, 1),
        )
        self.shortcut = nn.Identity() if channels == dim else nn.Conv2d(channels, dim, 1, bias=False)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(channels, max(channels // reduction, 1), 1),
                                 nn.ReLU(), nn.Conv2d(max(channels // reduction, 1), channels, 1))

    def forward(self, x):
        scale = self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True))
        return x * scale.sigmoid()


class AttentionGate(nn.Module):
    def __init__(self, channels, inter_channels):
        super().__init__()
        self.theta = nn.Conv2d(channels, inter_channels, 1, bias=False)
        self.phi = nn.Conv2d(channels, inter_channels, 1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, 1)

    def forward(self, x, g):
        return x * self.psi(torch.relu(self.theta(x) + self.phi(g))).sigmoid()


class ResonanceModel(nn.Module):
    def __init__(self, input_shape=(128, 256, 2)):
        super().__init__()
        if len(input_shape) != 3 or input_shape[2] != 2 or any(n <= 0 or n % 16 for n in input_shape[:2]):
            raise ValueError("input_shape must be (height, width, 2), with positive multiples of 16")
        self.input_shape = tuple(input_shape)
        self.encoder = nn.ModuleList()
        for cin, cout, kernel in ((2, 32, 4), (32, 64, 2), (64, 128, 2)):
            self.encoder.append(nn.Sequential(nn.Conv2d(cin, cout, kernel, stride=2, padding=1 if kernel == 4 else 0),
                                              ChannelNorm(cout), ConvNeXtBlock(cout, cout)))
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 2, stride=2), ChannelNorm(256),
                                        ConvNeXtBlock(256, 256), ConvNeXtBlock(256, 256), ChannelAttention(256))
        self.ups = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for cin, cout in ((256, 128), (128, 64), (64, 32)):
            self.ups.append(nn.Sequential(nn.ConvTranspose2d(cin, cout, 2, stride=2), ChannelNorm(cout)))
            self.gates.append(AttentionGate(cout, cout // 2))
            self.decoder.append(ConvNeXtBlock(cout * 2, cout))
        self.final = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), ChannelNorm(16), nn.GELU())
        self.output = nn.Conv2d(16, 2, 1)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (2, *self.input_shape[:2]):
            raise ValueError(f"Expected (N, 2, {self.input_shape[0]}, {self.input_shape[1]}), got {tuple(x.shape)}")
        skips = []
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)
        x = self.bottleneck(x)
        for up, gate, decoder, skip in zip(self.ups, self.gates, self.decoder, reversed(skips)):
            x = up(x)
            x = decoder(torch.cat((x, gate(skip, x)), dim=1))
        x = self.final(x)
        # Keep the final reconstruction and small normalized CSI losses in float32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.output(x.float())


def build_resonance_model(input_shape=(128, 256, 2)):
    return ResonanceModel(input_shape)


def nmse_metric(y_true, y_pred):
    y_true, y_pred = y_true.float(), y_pred.float()
    return (y_true - y_pred).square().mean() / (y_true.square().mean() + 1e-8)


def nmse_db_metric(y_true, y_pred):
    return 10.0 * nmse_metric(y_true, y_pred).clamp_min(1e-10).log10()


def resonance_loss(lambda_mag=0.05):
    def loss(y_true, y_pred):
        y_true, y_pred = y_true.float(), y_pred.float()
        true_mag = (y_true.square().sum(dim=1) + 1e-8).sqrt()
        pred_mag = (y_pred.square().sum(dim=1) + 1e-8).sqrt()
        mag_loss = (true_mag - pred_mag).square().mean() / (true_mag.square().mean() + 1e-8)
        return nmse_metric(y_true, y_pred) + lambda_mag * mag_loss
    return loss


if __name__ == "__main__":
    model = build_resonance_model()
    dummy = torch.randn(1, 2, 128, 256)
    out = model(dummy)
    assert out.shape == dummy.shape and torch.isfinite(out).all()
    resonance_loss()(dummy, out).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"Forward/backward passed; parameters: {sum(p.numel() for p in model.parameters()):,}")
