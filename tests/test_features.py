import unittest

import numpy as np

from etch.features import clamp_box, safe_zscore, temporal_autocorr


class FeatureTests(unittest.TestCase):
    def test_clamp_box_keeps_valid_one_pixel_box(self):
        self.assertEqual(clamp_box(-5, -2, -1, 0, 10, 8), (0, 0, 1, 1))
        self.assertEqual(clamp_box(9.7, 7.9, 50, 50, 10, 8), (9, 7, 10, 8))

    def test_safe_zscore_handles_constant_and_nan_values(self):
        np.testing.assert_array_equal(safe_zscore([3, 3, 3]), np.zeros(3, dtype=np.float32))
        z = safe_zscore([1, np.nan, 3])
        self.assertTrue(np.isfinite(z).all())
        self.assertAlmostEqual(float(z[1]), 0.0, places=6)

    def test_temporal_autocorr(self):
        a = np.arange(9, dtype=np.uint8).reshape(3, 3)
        b = a.copy()
        self.assertAlmostEqual(temporal_autocorr(a, b), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

