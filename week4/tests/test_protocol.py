import math
import unittest

from week4.common.protocol import decode_frame, pack_frame, validate_frame


class ProtocolTests(unittest.TestCase):
    def test_frame_structure_and_round_trip(self):
        source = [0.0, math.radians(30), math.radians(-45), 0.1, -0.2, 0.0]
        frame = pack_frame(source)
        validate_frame(frame)
        self.assertEqual(len(frame), 16)
        self.assertEqual(frame[0], 0xAA)
        self.assertEqual(frame[13], 0x01)
        self.assertEqual(frame[15], 0xBB)
        decoded = decode_frame(frame)
        for actual, expected in zip(decoded, source):
            self.assertAlmostEqual(actual, int(expected * 1000) / 1000, places=12)

    def test_checksum_failure_is_detected(self):
        frame = bytearray(pack_frame([0.0] * 6))
        frame[2] ^= 0x01
        with self.assertRaisesRegex(ValueError, "校验"):
            validate_frame(frame)


if __name__ == "__main__":
    unittest.main()

