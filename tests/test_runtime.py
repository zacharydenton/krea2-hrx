"""Regression checks for cache publication and the Python pipeline/session API."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import krea2_loom as native_api
from scripts import build_kernels as builder
from tools.pipeline import ReferenceForward, cast_transformer_bf16
from loom_ref import attention


class AttentionTests(unittest.TestCase):
    def test_default_and_16_use_unquantized_gqa(self):
        generator = torch.Generator().manual_seed(19)
        q = torch.randn(2, 65, 4, 128, generator=generator).half()
        k = torch.randn(2, 65, 1, 128, generator=generator).half()
        v = torch.randn(2, 65, 1, 128, generator=generator).half()
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float().transpose(1, 2),
            k.float().transpose(1, 2).repeat_interleave(4, 1),
            v.float().transpose(1, 2).repeat_interleave(4, 1)).transpose(1, 2)
        for choice in ("", "16"):
            with patch.dict(os.environ, {"KREA2_ATTN_QK": choice}):
                torch.testing.assert_close(attention(q, k, v), expected)
        for choice in ("4", "8"):
            with patch.dict(os.environ, {"KREA2_ATTN_QK": choice}):
                result = attention(q, k, v)
                self.assertTrue(torch.isfinite(result).all())
                self.assertFalse(torch.allclose(result, expected))
        with patch.dict(os.environ, {"KREA2_ATTN_QK": "12"}):
            with self.assertRaisesRegex(ValueError, "4, 8 or 16"):
                attention(q, k, v)


class CacheTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"KREA2_ATTN_QK": ""})
        env.start()
        self.addCleanup(env.stop)

    def test_config_source_changes_and_failed_publication(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shutil.copytree(ROOT / "kernels", root / "kernels")
            compiler = root / "compiler"
            compiler.write_text("test compiler")
            def compile_one(src, symbol, out, cfg):
                out.write_text(repr((src, symbol, cfg)))
            with patch.object(builder, "ROOT", root), patch.object(builder, "compiler", return_value=compiler), \
                 patch.object(builder, "compile_one", side_effect=compile_one) as compile_mock:
                first = builder.build(129)
                self.assertEqual(builder.build(129), first)
                self.assertEqual(compile_mock.call_count, 9)
                wide = builder.build(4115, 8)
                self.assertEqual((wide / "launch.txt").read_text(), "4 4115 256 4 4160 8 6144 16448 16 8\n")
                self.assertIn("gemm_i8_resid_256", (wide / "gemm_down.hsaco").read_text())
                self.assertIn("prepare_plain_i8", (wide / "prepare_plain_i8.hsaco").read_text())
                for tokens, waves in ((129, 8), (8191, 8), (8192, 4), (16896, 4)):
                    selected = builder.build(tokens)
                    launch = (selected / "launch.txt").read_text()
                    fields = launch.split()
                    self.assertEqual(fields[0], "4")
                    self.assertEqual(fields[5], str(waves))
                    self.assertEqual(fields[2], str(builder.gemm_rows(tokens)))
                    self.assertEqual(fields[6:], ["6144", "16512", "16", "4"])
                    self.assertIn("attention_gqa_lds_f16_wmma", (selected / "attention.hsaco").read_text())
                    self.assertEqual(builder.build(tokens), selected)
                    (selected / "launch.txt").write_text(launch.rsplit(" ", 1)[0] + " 9\n")
                    with self.assertRaisesRegex(RuntimeError, "launch metadata"):
                        builder.build(tokens)
                    (selected / "launch.txt").write_text(launch)
                # The int4/int8 QK kernels stay selectable, prefetching above 8192 tokens.
                for tokens, waves in ((129, 8), (8192, 4)):
                    with patch.dict(os.environ, {"KREA2_ATTN_QK": "4"}):
                        selected = builder.build(tokens)
                    expected = "attention_sage_i4_fast" + ("_prefetch" if waves == 4 else "")
                    self.assertIn(expected, (selected / "attention.hsaco").read_text())
                    self.assertEqual((selected / "launch.txt").read_text().split()[8], "4")
                source = root / "kernels/gemm_i4.loom"
                source.write_text(source.read_text() + "\n// source changed\n")
                before = set(first.parent.iterdir())
                with patch.object(builder, "compile_one", side_effect=RuntimeError("compile failed")):
                    with self.assertRaisesRegex(RuntimeError, "compile failed"):
                        builder.build(129)
                self.assertEqual(set(first.parent.iterdir()), before)
                third = builder.build(129)
                self.assertNotEqual(third, first)
                (third / "gemm_qkvg.hsaco").write_text("corrupted")
                with self.assertRaisesRegex(RuntimeError, "corrupt cached kernel"):
                    builder.build(129)

    def test_automatic_groups(self):
        for tokens, group in ((16, 1), (129, 2), (4115, 3), (8192, 4)):
            self.assertEqual(builder.gemm_m_group(tokens, 128), group)
        for tokens in (4096, 4115, 16896):
            self.assertEqual(builder.gemm_m_group(tokens, 256), 4)
        self.assertEqual([builder.gemm_pitch(k) for k in (6144, 16384)], [6144, 16512])
        self.assertEqual([builder.gemm_pitch(k, 8) for k in (6144, 16384)], [6144, 16448])
        self.assertEqual([builder.gemm_rows(t, 8) for t in (16, 1040, 4115)], [256, 256, 256])
        self.assertEqual([builder.gemm_rows(t) for t in (16, 1040, 2047, 2064, 4115, 4353, 8192, 16896)],
                         [128, 128, 128, 256, 256, 256, 256, 256])

    def test_reject_unsupported_tokens(self):
        for tokens in (0, 15, 16897, 65536):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, "16..16896"):
                builder.build(tokens)


class WrapperTests(unittest.TestCase):
    def test_forward_does_not_alias_input(self):
        class Native:
            def krea2_run(self, handle, x, count, *args):
                # The residual stream is bf16: add one in the buffer's own dtype.
                buffer = np.ctypeslib.as_array(x, shape=(count,))
                values = torch.from_numpy(buffer.view(np.int16)).view(torch.bfloat16) + 1
                buffer[:] = values.view(torch.int16).numpy().view(np.uint16)
                return 0
        block = native_api.Krea2Blocks.__new__(native_api.Krea2Blocks)
        block.tokens, block.layers, block._handle, block._native = 16, 1, None, Native()
        for dtype in (torch.float16, torch.float32):
            x = torch.ones(16, 6144, dtype=dtype)
            args = (torch.zeros(1, 6, 6144), torch.ones(16, 128), torch.zeros(16, 128))
            first, second = block.forward(x, *args), block.forward(x, *args)
            self.assertTrue(torch.equal(x, torch.ones_like(x)))
            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.equal(first, torch.full_like(first, 2)))
        for first, count in ((-1, 1), (1, 1), (0, 0), (0, 2)):
            with self.subTest(first=first, count=count), self.assertRaises(ValueError):
                block.forward(x, *args, first_block=first, block_count=count)


class AdapterTests(unittest.TestCase):
    def test_batched_masks_modulation_and_session_reuse(self):
        class Ref:
            layers = 1
            def text_in(self, x): return x
            def image_in(self, x): return x
            def time_embed(self, t): return t, t
            def block_modulation(self, i, m): return m[:, None, None, None].expand(-1, 1, 6, 6144)
            def final(self, x, t): return x
        sessions = []
        class Blocks:
            def __init__(self, tokens, layers, weights=None):
                self.tokens, self.closed = tokens, False
                sessions.append(self)
            def close(self): self.closed = True
            def forward(self, x, mods, cos, sin):
                assert not self.closed and len(x) == self.tokens
                return x + x[0] + mods[0, 0]
        adapter = ReferenceForward(Ref(), None, "loom")
        image = torch.zeros(2, 16, 6144)
        text = torch.zeros(2, 2, 6144)
        text[0, 0], text[1, 1] = 2, 5
        mask = torch.tensor([[1, 0], [0, 1]])
        positions = torch.tensor([[0, 0, 0], [0, 3, 3]])
        with patch.object(native_api, "Krea2Blocks", Blocks):
            out = adapter(image, text, torch.tensor([3, 7]), positions, mask).sample
            self.assertEqual(tuple(out.shape), tuple(image.shape))
            self.assertTrue(torch.all(out[0] == 5))
            self.assertTrue(torch.all(out[1] == 12))
            self.assertEqual(len(sessions), 1)
            # A new prompt length must close the old session and construct a new one.
            adapter(image[:1], text[:1], torch.tensor([3]), positions, torch.ones(1, 2))
            self.assertEqual(len(sessions), 2)
            self.assertTrue(sessions[0].closed)
            self.assertEqual(sessions[1].tokens, 18)

    def test_norm_precision_survives_manual_loading(self):
        from diffusers.models.transformers.transformer_krea2 import Krea2RMSNorm
        class Model(torch.nn.Module):
            _keep_in_fp32_modules = ["norm"]
            def __init__(self):
                super().__init__()
                self.norm = Krea2RMSNorm(128)
                self.linear = torch.nn.Linear(8, 8)
        model = Model()
        with torch.no_grad():
            model.norm.weight.fill_(0.002001234)
        original = model.norm.weight.detach().clone()
        cast_transformer_bf16(model)
        self.assertEqual(model.linear.weight.dtype, torch.bfloat16)
        self.assertEqual(model.norm.weight.dtype, torch.float32)
        self.assertTrue(torch.equal(model.norm.weight, original))
        x = torch.linspace(-2, 2, 256).reshape(2, 128).bfloat16()
        expected = torch.nn.functional.rms_norm(x.float(), (128,), weight=original + 1, eps=model.norm.eps).bfloat16()
        self.assertTrue(torch.equal(model.norm(x), expected))


class WorkflowTests(unittest.TestCase):
    def test_failed_children_do_not_publish_completion(self):
        workflows = {"quality.sh": "build/quality.done", "quality_rest.sh": "build/quality.done",
                     "quality_then_profile.sh": "build/profile.done", "e2e_loom.sh": "build/e2e.done",
                     "after_baseline.sh": "build/export.done"}
        for workflow, marker in workflows.items():
            with self.subTest(workflow=workflow), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                shutil.copytree(ROOT / "scripts", root / "scripts")
                bin_dir = root / ".venv/bin"
                bin_dir.mkdir(parents=True)
                (bin_dir / "activate").write_text("")
                for name, status in (("python3", 17), ("pgrep", 1)):
                    executable = bin_dir / name
                    executable.write_text(f"#!/bin/sh\nexit {status}\n")
                    executable.chmod(0o755)
                (root / "build").mkdir(exist_ok=True)
                (root / marker).write_text("stale success")
                result = subprocess.run(["bash", str(root / "scripts" / workflow)],
                                        env=dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"]),
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 17, result.stderr)
                self.assertFalse((root / marker).exists())


if __name__ == "__main__":
    unittest.main()
