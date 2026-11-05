"""TinyML CSI Denoiser for ESP32: Architecture, Training, and Embedded C Code Generation."""

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from wifi_dataset import generate_wifi_iot_split, WIFI_SCENARIOS, WiFiScenarioSpec


class TinyWiFiDenoiser(nn.Module):
    """
    Ultra-compact 1D depthwise-separable CNN denoiser tailored for ESP32 SRAM and flash constraints.
    - Total parameters: ~2,690
    - Model flash footprint: ~10.7 KB (Float32) / ~2.7 KB (INT8)
    - SRAM workspace: < 16 KB
    - Compute complexity: ~170k FLOPs (< 1 ms @ 240 MHz)
    """

    def __init__(self, subcarriers=64):
        super().__init__()
        self.subcarriers = subcarriers

        # 1. Input projection: (2, 64) -> (16, 64)
        self.in_conv = nn.Conv1d(2, 16, kernel_size=1)
        self.relu1 = nn.ReLU()

        # 2. Block 1: Depthwise 5 + Pointwise 16 -> 32
        self.dw1 = nn.Conv1d(16, 16, kernel_size=5, padding=2, groups=16)
        self.pw1 = nn.Conv1d(16, 32, kernel_size=1)
        self.relu2 = nn.ReLU()

        # 3. Block 2: Depthwise 5 + Pointwise 32 -> 32
        self.dw2 = nn.Conv1d(32, 32, kernel_size=5, padding=2, groups=32)
        self.pw2 = nn.Conv1d(32, 32, kernel_size=1)
        self.relu3 = nn.ReLU()

        # 4. Block 3: Depthwise 5 + Pointwise 32 -> 16
        self.dw3 = nn.Conv1d(32, 32, kernel_size=5, padding=2, groups=32)
        self.pw3 = nn.Conv1d(32, 16, kernel_size=1)
        self.relu4 = nn.ReLU()

        # 5. Output projection: 16 -> 2
        self.out_conv = nn.Conv1d(16, 2, kernel_size=1)

        # Initialize output to zero for safe identity baseline
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        res = self.in_conv(x)
        res = self.relu1(res)

        res = self.dw1(res)
        res = self.pw1(res)
        res = self.relu2(res)

        res = self.dw2(res)
        res = self.pw2(res)
        res = self.relu3(res)

        res = self.dw3(res)
        res = self.pw3(res)
        res = self.relu4(res)

        correction = self.out_conv(res)
        return x + correction


def train_tinyml_model(epochs=15, batch_size=32, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n" + "=" * 78)
    print("      TINYML MODEL FOR ESP32: TRAINING & ARCHITECTURE VERIFICATION")
    print("=" * 78)

    # 1. Dataset Generation
    scenarios = [
        WiFiScenarioSpec(s.name, s.delay_spread_ns, s.rician_k_db, s.max_doppler_hz, s.sample_count)
        for s in WIFI_SCENARIOS
    ]
    train_ds, val_ds, test_ds = generate_wifi_iot_split(scenarios=scenarios, seed=seed, clean_fraction=0.50)

    x_train = torch.from_numpy(train_ds.samples).permute(0, 2, 1).float()
    y_train = torch.from_numpy(train_ds.clean_samples).permute(0, 2, 1).float()
    x_val = torch.from_numpy(val_ds.samples).permute(0, 2, 1).float()
    y_val = torch.from_numpy(val_ds.clean_samples).permute(0, 2, 1).float()
    snr_val = torch.from_numpy(val_ds.snr_db).float()

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    model = TinyWiFiDenoiser(subcarriers=64)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"TinyWiFiDenoiser Parameters: {param_count} ({param_count * 4 / 1024:.2f} KB float32, {param_count / 1024:.2f} KB int8)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    print(f"Training on {len(x_train)} samples across {epochs} epochs...")
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(bx)
        scheduler.step()
        ep_loss /= len(x_train)
        if ep % 5 == 0 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}: Train MSE = {ep_loss:.6f}")

    # Validation
    model.eval()
    print("\n--- Multi-SNR Validation Scoreboard ---")
    metrics = {}
    for snr in [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
        mask = (snr_val == snr)
        xv = x_val[mask]
        yv = y_val[mask]
        with torch.no_grad():
            pred = model(xv)
        noisy_nmse = torch.mean(torch.sum((xv - yv) ** 2, dim=(1, 2)) / torch.sum(yv ** 2, dim=(1, 2))).item()
        pred_nmse = torch.mean(torch.sum((pred - yv) ** 2, dim=(1, 2)) / torch.sum(yv ** 2, dim=(1, 2))).item()
        gain_db = 10 * np.log10(noisy_nmse / pred_nmse)
        metrics[str(snr)] = {
            "noisy_nmse_db": 10 * np.log10(noisy_nmse),
            "pred_nmse_db": 10 * np.log10(pred_nmse),
            "gain_db": gain_db,
        }
        print(f"  SNR {snr:5.1f} dB: Noisy = {10*np.log10(noisy_nmse):6.2f} dB, Denoised = {10*np.log10(pred_nmse):6.2f} dB, NMSE Gain = {gain_db:+5.2f} dB")

    # Save model checkpoint
    out_dir = ROOT / "output/wifi_training/tinyml_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "tinyml_weights.pt"
    torch.save(model.state_dict(), weights_path)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVED] PyTorch weights saved to {weights_path}")

    return model, metrics


def export_to_embedded_c(model):
    """Generates zero-dependency embedded C source and weights header for ESP32."""
    model.eval()
    state = model.state_dict()

    out_src = ROOT / "hardware/esp32/securelink_c/src"
    out_inc = ROOT / "hardware/esp32/securelink_c/include"

    weights_header = out_inc / "securelink_tinyml_weights.h"

    def format_array(name, tensor):
        arr = tensor.cpu().detach().numpy().flatten()
        lines = [f"static const float {name}[{len(arr)}] = {{"]
        chunk_size = 8
        for i in range(0, len(arr), chunk_size):
            chunk = arr[i:i + chunk_size]
            vals = ", ".join(f"{x:.8f}f" for x in chunk)
            lines.append(f"    {vals},")
        lines.append("};\n")
        return "\n".join(lines)

    h_content = [
        "/**",
        " * Autogenerated TinyML weights header for ESP32 CSI denoiser.",
        " * Zero-dependency static constants compiled into flash.",
        " */",
        "#ifndef SECURELINK_TINYML_WEIGHTS_H",
        "#define SECURELINK_TINYML_WEIGHTS_H\n",
        format_array("tinyml_in_conv_w", state["in_conv.weight"]),
        format_array("tinyml_in_conv_b", state["in_conv.bias"]),
        format_array("tinyml_dw1_w", state["dw1.weight"]),
        format_array("tinyml_dw1_b", state["dw1.bias"]),
        format_array("tinyml_pw1_w", state["pw1.weight"]),
        format_array("tinyml_pw1_b", state["pw1.bias"]),
        format_array("tinyml_dw2_w", state["dw2.weight"]),
        format_array("tinyml_dw2_b", state["dw2.bias"]),
        format_array("tinyml_pw2_w", state["pw2.weight"]),
        format_array("tinyml_pw2_b", state["pw2.bias"]),
        format_array("tinyml_dw3_w", state["dw3.weight"]),
        format_array("tinyml_dw3_b", state["dw3.bias"]),
        format_array("tinyml_pw3_w", state["pw3.weight"]),
        format_array("tinyml_pw3_b", state["pw3.bias"]),
        format_array("tinyml_out_conv_w", state["out_conv.weight"]),
        format_array("tinyml_out_conv_b", state["out_conv.bias"]),
        "#endif /* SECURELINK_TINYML_WEIGHTS_H */\n"
    ]

    with open(weights_header, "w", encoding="utf-8") as f:
        f.write("\n".join(h_content))
    with open(out_src / "securelink_tinyml_weights.h", "w", encoding="utf-8") as f:
        f.write("\n".join(h_content))

    print(f"[EXPORT] Generated weights header at {weights_header}")


if __name__ == "__main__":
    model, metrics = train_tinyml_model(epochs=15)
    export_to_embedded_c(model)
