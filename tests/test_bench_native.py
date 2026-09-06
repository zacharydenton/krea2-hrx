"""Exercise repeat-image failures without loading the model or a GPU runtime."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import bench_native


class RepeatImageTests(unittest.TestCase):
    def run_benchmark(self, root, outputs):
        library = Mock()
        library.krea2_pipeline_create.return_value = 0
        images = iter(outputs)

        def generate(*args):
            # The C API overwrites the same caller-owned buffer on every run.
            args[8][:] = next(images)
            return 0

        library.krea2_generate.side_effect = generate
        argv = ["bench_native", "--size", "64", "--runs", str(len(outputs))]
        try:
            with (
                patch.object(bench_native, "ROOT", root),
                patch.object(bench_native.C, "CDLL", return_value=library),
                patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                bench_native.main()
        finally:
            library.krea2_pipeline_destroy.assert_called_once()

    def test_late_mismatch_preserves_images_and_pixel_errors(self):
        first = bytes([20]) * (64 * 64 * 3)
        changed = bytearray(first)
        changed[0], changed[1], changed[6] = 21, 18, 24
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "images and error report"):
                self.run_benchmark(root, [first, first, bytes(changed)])
            (output,) = (root / "build").glob("native-mismatch-*")
            header = b"P6\n64 64\n255\n"
            self.assertEqual((output / "first.ppm").read_bytes(), header + first)
            self.assertEqual((output / "repeat.ppm").read_bytes(), header + changed)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["repeat_run"], 2)
            self.assertEqual(report["changed_channels"], 3)
            self.assertEqual(report["changed_pixels"], 2)
            self.assertEqual(report["max_absolute_error"], 4)
            self.assertEqual(report["mean_absolute_error"], 7 / len(first))
            self.assertNotEqual(report["first_rgb_sha256"], report["repeat_rgb_sha256"])

    def test_equal_repeats_do_not_create_failure_artifacts(self):
        first = bytes(64 * 64 * 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_benchmark(root, [first, first, first])
            self.assertFalse((root / "build").exists())


if __name__ == "__main__":
    unittest.main()
