"""Compare native Turbo/Raw sigma grids with diffusers entirely on the CPU."""
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.pipeline import SCHEDULER


class ScheduleTests(unittest.TestCase):
    def test_sigma_grids_match_diffusers(self):
        # 589 tokens (304x496) exposes the early float32 rounding of Raw mu.
        image_tokens = (0, 16, 256, 589, 4096, 6400, 16384)
        native = np.frombuffer(
            subprocess.check_output([str(ROOT / "build/krea2-schedule-grid")]), dtype=np.float32)
        offset = 0
        for tokens in image_tokens:
            m = (1.15 - 0.5) / (6400 - 256)
            mu = tokens * m + (0.5 - m * 256) if tokens else 1.15
            for steps in range(1, 101):
                scheduler = FlowMatchEulerDiscreteScheduler(**SCHEDULER)
                scheduler.set_timesteps(sigmas=np.linspace(1., 1. / steps, steps), mu=mu, device="cpu")
                np.testing.assert_array_equal(native[offset:offset + steps + 1], scheduler.sigmas.numpy(),
                                              err_msg=f"tokens={tokens}, steps={steps}")
                offset += steps + 1
        self.assertEqual(offset, native.size)


if __name__ == "__main__":
    unittest.main()
