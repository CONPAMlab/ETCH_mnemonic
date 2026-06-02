import tempfile
import unittest
import importlib.util
from argparse import Namespace
from pathlib import Path

import pandas as pd

from etch.cli import _load_predictive_train_args
from etch.schema import OBJECT_TRACK_COLUMNS


HAS_TORCH = importlib.util.find_spec("torch") is not None


class PredictiveModelConfigTests(unittest.TestCase):
    def test_train_predictive_config_is_loadable(self):
        args = Namespace(
            config="configs/predictive_model.yaml",
            features=None,
            output=None,
            sequence_length=8,
            prediction_horizon=1,
            hidden_dim=128,
            latent_dim=64,
            batch_size=64,
            epochs=5,
            learning_rate=1e-3,
            min_track_length=12,
            device="cpu",
            seed=123,
        )
        values = _load_predictive_train_args(args)
        self.assertEqual(values["features"], "runs/ego_nvs_smoke")
        self.assertEqual(values["output"], "runs/ego_nvs_smoke/predictive_model")
        self.assertEqual(values["sequence_length"], 8)


@unittest.skipUnless(HAS_TORCH, "torch is required for predictive-model training")
class PredictiveModelTests(unittest.TestCase):
    def test_train_predictive_model_on_synthetic_tracks(self):
        from etch.predictive_model import (
            OBJECT_STATE_COLUMNS,
            PredictiveTrainingConfig,
            train_predictive_model,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features" / "object_tracks"
            feature_dir.mkdir(parents=True)
            rows = []
            for track_id in [1, 2]:
                for frame in range(10):
                    row = {c: 0.0 for c in OBJECT_TRACK_COLUMNS}
                    row.update(
                        {
                            "video_uid": "synthetic",
                            "video_path": "synthetic.mp4",
                            "frame": frame,
                            "track_id": track_id,
                            "cls_name": "cup",
                            "cx_norm": 0.1 * track_id + frame * 0.01,
                            "cy_norm": 0.2 * track_id + frame * 0.01,
                            "area_rel": 0.02 + frame * 0.001,
                            "mean_r": 100 + frame,
                            "mean_g": 90 + frame,
                            "mean_b": 80 + frame,
                            "mean_s": 30 + frame,
                            "mean_v": 120 + frame,
                            "speed_px_s": 5 + frame,
                            "saliency_score": 0.2 + frame * 0.01,
                            "temporal_rgb_drift": 1 + frame * 0.1,
                            "feature_pred_err": 0.5 + frame * 0.05,
                        }
                    )
                    rows.append(row)
            pd.DataFrame(rows).to_parquet(feature_dir / "synthetic.parquet", index=False)
            report = train_predictive_model(
                root,
                root / "predictive_model",
                PredictiveTrainingConfig(
                    sequence_length=3,
                    prediction_horizon=1,
                    hidden_dim=16,
                    latent_dim=8,
                    batch_size=4,
                    epochs=1,
                    min_track_length=4,
                    device="cpu",
                ),
            )
            self.assertGreater(report["n_windows"], 0)
            self.assertTrue((root / "predictive_model" / "object_predictive_coding.pt").exists())
            self.assertTrue((root / "predictive_model" / "predictive_memory_alignment.parquet").exists())
            self.assertEqual(report["n_features"], len(OBJECT_STATE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
