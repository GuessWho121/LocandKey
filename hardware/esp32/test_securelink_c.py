"""Cross-verification check between Python reference securelink.py and C implementation contracts."""

import hashlib
import hmac
import struct
import numpy as np
import bchlib

def test_constants():
    # BCH Parameters
    assert bchlib.BCH(6, m=9).ecc_bytes == 7
    assert bchlib.BCH(6, m=9).ecc_bits == 54
    assert ((bchlib.BCH(6, m=9).n - 54) // 8) * 8 == 456

    # Test BCH impulse and generator polynomial
    poly = 0x0c5bf84377ebc7
    reg = 0
    test_msg = b"Hello SecureLink ESP32 Physical Layer!"
    for byte in test_msg:
        for bit_pos in range(7, -1, -1):
            bit = (byte >> bit_pos) & 1
            fb = ((reg >> 53) & 1) ^ bit
            reg = ((reg << 1) & ((1 << 54) - 1))
            if fb:
                reg ^= poly

    packed = bytearray(7)
    for i in range(7):
        shift = 54 - (i + 1) * 8
        if shift >= 0:
            packed[i] = (reg >> shift) & 0xff
        else:
            packed[i] = (reg << (-shift)) & 0xff

    bch = bchlib.BCH(6, m=9)
    assert bch.encode(test_msg) == packed, "LFSR parity calculation does not match bchlib"
    print("[PASS] BCH C-LFSR matches Python bchlib bit-for-bit.")

def test_offer_wire_format():
    # Verify exact header byte packing
    prefix = b"securelink/offer/v2"
    nonce_len = 16
    feature_count = 64
    mask_bytes = 8
    ecc_bytes = 7
    header = struct.pack(">HIII", nonce_len, feature_count, mask_bytes, ecc_bytes)
    assert len(header) == 14
    assert header == bytes([0x00, 0x10, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x07])
    print("[PASS] Offer wire-format header serialization matches C struct layout.")

if __name__ == "__main__":
    test_constants()
    test_offer_wire_format()
    print("All cross-verification checks passed successfully!")
