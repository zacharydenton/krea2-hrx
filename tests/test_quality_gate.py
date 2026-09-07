"""CPU-only checks that either trajectory metric can block a release."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.quality_vs_bf16 import quality_check


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        baseline, candidate = root / "baseline", root / "candidate"
        baseline.mkdir()
        (baseline / "job.json").write_text(json.dumps({"seed": 0, "steps": 8}))
        for name in ("noise.npy", "text.npy"):
            np.save(baseline / name, np.ones((4, 4), dtype=np.float32))
        torch.save(torch.ones(1, 16, 64), baseline / "bf16.pt")
        torch.save(torch.ones(1, 16, 64) + .01, baseline / "w8a8.pt")
        for name, value in (("bf16.png", 128), ("w8a8.png", 129)):
            Image.fromarray(np.full((16, 16, 3), value, np.uint8)).save(baseline / name)
        shutil.copytree(baseline, candidate)
        self.args = SimpleNamespace(baseline=baseline, work=candidate, max_drop_db=.1)

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            quality_check(self.args)

    def test_equal_quality_passes(self):
        self.check()
        self.assertTrue(json.loads((self.args.work / "quality-check.json").read_text())["passed"])

    def test_latent_regression_fails_even_with_identical_images(self):
        # Constant scaling keeps cosine at 1 while increasing trajectory error.
        torch.save(torch.ones(1, 16, 64) + .02, self.args.work / "w8a8.pt")
        with self.assertRaisesRegex(SystemExit, "quality regression"):
            self.check()

    def test_image_regression_fails_even_with_identical_latents(self):
        Image.fromarray(np.full((16, 16, 3), 130, np.uint8)).save(self.args.work / "w8a8.png")
        with self.assertRaisesRegex(SystemExit, "quality regression"):
            self.check()

    def test_different_inputs_cannot_pass(self):
        np.save(self.args.work / "noise.npy", np.zeros((4, 4), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "identical noise"):
            self.check()

    def test_nonfinite_latents_cannot_pass(self):
        torch.save(torch.full((1, 16, 64), float("nan")), self.args.work / "w8a8.pt")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.check()


if __name__ == "__main__":
    unittest.main()
