import tempfile
import unittest
from pathlib import Path

import pandas as pd

from etch.config import load_config
from etch.io import assert_output_not_inside_dataset
from etch.manifest import discover_videos


class ConfigManifestTests(unittest.TestCase):
    def test_config_loads_relative_paths_and_manifest_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_dir = root / "full_scale"
            video_dir.mkdir()
            for uid in ["a", "b", "c"]:
                (video_dir / f"{uid}.mp4").touch()
            pd.DataFrame(
                [
                    {"video_uid": "a", "scenarios": "kitchen", "split_fho": "train"},
                    {"video_uid": "b", "scenarios": "driving", "split_fho": "train"},
                    {"video_uid": "c", "scenarios": "kitchen", "split_fho": "val"},
                ]
            ).to_csv(video_dir / "manifest.csv", index=False)
            cfg = root / "config.yaml"
            cfg.write_text(
                """
dataset:
  root: .
  video_dir: full_scale
  manifest: full_scale/manifest.csv
run:
  output_root: out
sample:
  n_videos: 1
  seed: 7
  scenario_contains: kitchen
model:
  device: cpu
""",
                encoding="utf-8",
            )
            config = load_config(cfg)
            manifest = discover_videos(config)
            self.assertEqual(len(manifest), 1)
            self.assertIn(manifest.iloc[0]["video_uid"], {"a", "c"})
            self.assertTrue(str(config.run.output_root).endswith("out"))

    def test_refuses_output_inside_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                assert_output_not_inside_dataset(root / "runs", root)


if __name__ == "__main__":
    unittest.main()

