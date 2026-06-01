import tempfile
import unittest
from pathlib import Path

import pandas as pd

from etch.predictive import build_predictive_scaffold
from etch.schema import OBJECT_TRACK_COLUMNS, missing_columns


class SchemaPredictiveTests(unittest.TestCase):
    def test_missing_columns(self):
        self.assertEqual(missing_columns(["a", "b"], ["a", "c"]), ["c"])

    def test_predictive_scaffold_from_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "objects.parquet"
            rows = []
            for i in range(5):
                row = {c: None for c in OBJECT_TRACK_COLUMNS}
                row.update(
                    {
                        "video_uid": "v",
                        "frame": i,
                        "track_id": 1,
                        "cls_name": "cup",
                        "feature_pred_err": float(i),
                        "traj_pred_err_px": float(i + 1),
                        "temporal_rgb_drift": float(i + 2),
                        "area_rel": 0.01 * (i + 1),
                        "mean_s": 20.0 + i,
                        "novelty_score": float(i - 2),
                    }
                )
                rows.append(row)
            pd.DataFrame(rows).to_parquet(path, index=False)
            report = build_predictive_scaffold(path, Path(tmp) / "out")
            self.assertEqual(report["n_rows"], 5)
            self.assertTrue((Path(tmp) / "out" / "study4b_predictive_targets.parquet").exists())


if __name__ == "__main__":
    unittest.main()

