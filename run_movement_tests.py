"""
Unified Entry Point for SecureLink Movement & Doppler Test Suite.

Executes:
1. Suite 1: Physical Doppler Velocity & Movement Spectrum Sweep (0 to 50 Hz)
2. Suite 2: Reciprocity Delay Breakdown vs. Coherence Window (0.5 to 20 ms)
3. Suite 3 (Optional): Live Physical ESP32 Hardware Movement Diagnostic
"""

import sys
from hardware.esp32.movement_suite.run_all import main

if __name__ == "__main__":
    sys.exit(main())
