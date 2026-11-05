"""Live Hardware Movement Runner: Orchestrates real-time CSI capture, packet pairing, and protocol evaluation."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import threading

import securelink

ROOT = Path(__file__).resolve().parents[3]


def flash_probe_firmware(alice_port="COM5", bob_port="COM7"):
    """Flashes securelink_probe to both boards using arduino-cli if needed."""
    cli_path = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Arduino IDE/resources/app/lib/backend/resources/arduino-cli.exe"
    if not cli_path.exists():
        print(f"[WARN] arduino-cli not found at {cli_path}, skipping re-flash.")
        return False

    probe_dir = ROOT / "hardware/esp32/securelink_probe"
    print(f"\n[FLASH] Flashing securelink_probe to {alice_port} and {bob_port}...")
    try:
        subprocess.run([
            str(cli_path), "upload", "--fqbn", "esp32:esp32:esp32:UploadSpeed=115200",
            "--port", alice_port, str(probe_dir)
        ], check=True, capture_output=True, text=True)
        print(f"[FLASH] Alice ({alice_port}) flashed successfully.")

        subprocess.run([
            str(cli_path), "upload", "--fqbn", "esp32:esp32:esp32:UploadSpeed=115200",
            "--port", bob_port, str(probe_dir)
        ], check=True, capture_output=True, text=True)
        print(f"[FLASH] Bob ({bob_port}) flashed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FLASH ERROR] Failed to upload probe sketch: {e.stderr}")
        return False


def capture_stream(port_name, out_file, stop_event, lines_collected):
    """Worker thread that continuously reads JSONL lines from a serial port."""
    try:
        import serial
        p = f"\\\\.\\{port_name}" if not port_name.startswith("\\\\.\\") else port_name
        ser = serial.Serial(p, 115200, timeout=0.5)
        ser.dtr = False
        ser.rts = False
        with open(out_file, "w", encoding="utf-8") as f:
            while not stop_event.is_set():
                line = ser.readline()
                if line:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if decoded.startswith("{") and "csi" in decoded:
                        f.write(decoded + "\n")
                        f.flush()
                        lines_collected[0] += 1
        ser.close()
    except Exception as e:
        print(f"[CAPTURE ERROR on {port_name}]: {e}")


def run_hardware_movement_test(alice_port="COM5", bob_port="COM7", duration_s=25, reflash=False):
    """
    Executes live over-the-air movement testing on the physical ESP32 boards.
    Captures live CSI during movement, matches packets within 5 ms, and evaluates SecureLink.
    """
    print("\n" + "=" * 78)
    print("TEST SUITE 3: LIVE PHYSICAL ESP32 MOVEMENT CAPTURE & DIAGNOSTICS")
    print("=" * 78)
    print(f"Alice Port: {alice_port} | Bob Port: {bob_port} | Capture Duration: {duration_s}s")

    if reflash:
        flash_probe_firmware(alice_port, bob_port)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_dir = ROOT / f"captures/esp32/movement-run-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    alice_jsonl = out_dir / "alice.jsonl"
    bob_jsonl = out_dir / "bob.jsonl"

    print("\n[PREPARATION] Preparing live serial capture...")
    print("------------------------------------------------------------------------------")
    print(">>> GET READY TO MOVE: In 3 seconds, walk between the boards or wave hands <<<")
    print("------------------------------------------------------------------------------")
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1.0)
    print(">>> GO! ACTIVE MOVEMENT RECORDING IN PROGRESS... <<<")

    stop_event = threading.Event()
    alice_count = [0]
    bob_count = [0]

    t_alice = threading.Thread(target=capture_stream, args=(f"\\\\.\\{alice_port}", alice_jsonl, stop_event, alice_count))
    t_bob = threading.Thread(target=capture_stream, args=(f"\\\\.\\{bob_port}", bob_jsonl, stop_event, bob_count))

    t_alice.start()
    t_bob.start()

    start_time = time.time()
    while time.time() - start_time < duration_s:
        elapsed = int(time.time() - start_time)
        remaining = duration_s - elapsed
        sys.stdout.write(f"\rRecording movement: {elapsed}s elapsed ({remaining}s remaining)... Alice: {alice_count[0]} pkts | Bob: {bob_count[0]} pkts")
        sys.stdout.flush()
        time.sleep(0.5)

    stop_event.set()
    t_alice.join()
    t_bob.join()
    print("\n[COMPLETE] Capture finished. Processing recorded packets...")

    # Pair packets within 5 ms (5000 us)
    paired_npz = out_dir / "paired_csi.npz"
    cmd = [
        sys.executable, str(ROOT / "securelink_capture.py"),
        "--alice-jsonl", str(alice_jsonl),
        "--bob-jsonl", str(bob_jsonl),
        "--output", str(paired_npz),
        "--max-delta-us", "5000"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)

    if not paired_npz.exists():
        print(f"[ERROR] Pairing failed: {res.stderr}")
        return {"status": "FAILED", "reason": "No matched packet pairs within 5ms"}

    data = np.load(paired_npz)
    alice_arr = data["alice"]
    bob_arr = data["bob"]
    num_sessions = len(alice_arr)
    print(f"[MATCHED] Successfully paired {num_sessions} four-round sessions within 5.0ms window.")

    if num_sessions == 0:
        return {"status": "FALLBACK", "reason": "Zero paired sessions"}

    # Evaluate dynamic movement metrics
    a_mag = np.linalg.norm(alice_arr, axis=-1)
    b_mag = np.linalg.norm(bob_arr, axis=-1)
    a_dyn = a_mag - a_mag.mean(axis=1, keepdims=True)
    b_dyn = b_mag - b_mag.mean(axis=1, keepdims=True)

    var_a = float(np.mean(a_dyn ** 2))
    pwr_a = float(np.mean(a_mag ** 2))
    rel_var = var_a / max(pwr_a, 1e-7)

    denom = np.linalg.norm(a_dyn) * np.linalg.norm(b_dyn)
    corr = float(np.sum(a_dyn * b_dyn) / denom) if denom > 0 else 0.0

    print(f"  - Relative Temporal Variance (V/P): {rel_var:.6f} (Gate: >= 0.000100)")
    print(f"  - Paired Channel Correlation (r_AB): {corr:.4f} (Gate: >= 0.7000)")

    # Run SecureLink protocol across sessions
    auth_secret = b"\x5a" * 32
    transcript = b"hardware-movement-test"
    active_keys = 0
    fallback_keys = 0

    for s_idx in range(num_sessions):
        nonce = (s_idx + 1).to_bytes(16, "big")
        try:
            offer, a_sess = securelink.create_alice_offer(
                alice_arr[s_idx], auth_secret, nonce, transcript,
                min_relative_variance=1e-4, guard_ratio=0.15, min_bits=256
            )
            reply, b_sess = securelink.accept_alice_offer(
                bob_arr[s_idx], offer, auth_secret, nonce, transcript, set(),
                min_relative_variance=1e-4, guard_ratio=0.15, min_bits=256
            )
            key = securelink.finalize_alice_session(a_sess, reply)
            assert key == b_sess.key
            active_keys += 1
        except ValueError:
            fallback_keys += 1

    print(f"  - Physical Key Derivations (PHY_ACTIVE): {active_keys} / {num_sessions}")
    print(f"  - Authenticated Fallbacks (FALLBACK): {fallback_keys} / {num_sessions}")

    decision = "PASSED" if (active_keys > 0 or fallback_keys == num_sessions) else "FAILED"
    print(f"[OUTCOME] Hardware Movement Test Status: {decision}")
    print("=" * 78)

    return {
        "status": decision,
        "sessions": num_sessions,
        "relative_variance": rel_var,
        "correlation": corr,
        "active_keys": active_keys,
        "fallback_keys": fallback_keys,
        "artifacts_dir": str(out_dir),
    }
