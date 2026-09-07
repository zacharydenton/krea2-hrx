"""CPU checks of the experimental gfx11 attention lane mapping and arithmetic.

These compare the proposed wave operations with independent dense attention.
They do not validate the compiler's machine code; GPU checks remain required.
"""
import unittest
from unittest import mock
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from bench_attention import prepare_inputs
import bench_attention


def result_lanes(matrix):
    """gfx11 wave32 f32 accumulator: lane = column + 16 * row parity."""
    return np.stack([matrix[lane // 16::2, lane % 16] for lane in range(32)])


def rhs_repack(lanes):
    """Model f16 conversion, half-wave exchange, then halfword selection."""
    halves = lanes.astype(np.float16)
    result = np.empty((32, 16), np.float16)
    for lane in range(32):
        parity = lane // 16
        result[lane, parity::2] = halves[lane]
        result[lane, 1 - parity::2] = halves[lane ^ 16]
    return result


def tree_sum(values):
    # Match the proposed three-level f32 sum rather than NumPy's reduction
    # order; dense_attention below remains an independent f64 oracle.
    values = values[:, 0::2] + values[:, 1::2]
    values = values[:, 0::2] + values[:, 1::2]
    return values[:, 0] + values[:, 1]


def query_attention(q, k, v, key_tile=16):
    assert key_tile in (16, 32, 48, 64)
    tokens, dim = q.shape
    key_tokens = k.shape[0]
    out = np.empty_like(q)
    for origin in range(0, tokens, 16):
        queries = np.zeros((16, dim), np.float32)
        present = min(16, tokens - origin)
        queries[:present] = q[origin:origin + present]
        maxima = np.full(32, -1e9, np.float32)
        sums = np.zeros(32, np.float32)
        accumulators = np.zeros((dim // 16, 32, 8), np.float32)
        for key_origin in range(0, key_tokens, key_tile):
            keys = np.zeros((key_tile, dim), np.float32)
            values = np.zeros_like(keys)
            count = min(key_tile, key_tokens - key_origin)
            keys[:count] = k[key_origin:key_origin + count]
            values[:count] = v[key_origin:key_origin + count]
            scores = (keys @ queries.T) * np.float32(dim ** -.5)
            scores[count:] = -1e9
            lanes = result_lanes(scores)
            local_max = lanes.max(axis=1)
            next_max = np.maximum(maxima, np.maximum(local_max, local_max[np.arange(32) ^ 16]))
            weights = np.exp(lanes - next_max[:, None])
            old_scale = np.exp(maxima - next_max)
            tile_sum = tree_sum(weights[:, :8])
            if key_tile == 64:
                tile_sum = ((tile_sum + tree_sum(weights[:, 8:16])) +
                            (tree_sum(weights[:, 16:24]) + tree_sum(weights[:, 24:32])))
            else:
                for half in range(1, key_tile // 16):
                    tile_sum += tree_sum(weights[:, half * 8:(half + 1) * 8])
            sums = sums * old_scale + tile_sum
            probabilities = [rhs_repack(weights[:, half * 8:(half + 1) * 8])[:16].astype(np.float32).T
                             for half in range(key_tile // 16)]
            for c in range(dim // 16):
                accumulators[c] *= old_scale[:, None]
                for half, probability in enumerate(probabilities):
                    pv = values[half * 16:(half + 1) * 16, c * 16:(c + 1) * 16].T @ probability
                    accumulators[c] += result_lanes(pv)
            maxima = next_max
        totals = sums + sums[np.arange(32) ^ 16]
        for c in range(dim // 16):
            dense = rhs_repack(accumulators[c] / totals[:, None])[:16]
            out[origin:origin + present, c * 16:(c + 1) * 16] = dense[:present]
    return out


def dense_attention(q, k, v):
    logits = q.astype(np.float64) @ k.astype(np.float64).T / np.sqrt(q.shape[1])
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    return weights @ v.astype(np.float64) / weights.sum(axis=1, keepdims=True)


class QueryAttentionTest(unittest.TestCase):
    def test_repack_all_rows_columns_and_halfwaves(self):
        matrix = np.arange(256, dtype=np.float32).reshape(16, 16)
        packed = rhs_repack(result_lanes(matrix))
        np.testing.assert_array_equal(packed[:16].T, matrix)
        np.testing.assert_array_equal(packed[:16], packed[16:])

    def test_dense_oracle_partial_tiles_and_sharp_softmax(self):
        random = np.random.default_rng(917)
        for tokens in (16, 17, 31, 32, 65, 129):
            for scale in (.5, 1., 3.):
                with self.subTest(tokens=tokens, scale=scale):
                    q, k = [(random.normal(size=(tokens, 128)) * scale).astype(np.float16)
                            for _ in range(2)]
                    v = random.normal(size=(tokens, 128)).astype(np.float16)
                    actual = query_attention(q, k, v).astype(np.float64)
                    expected = dense_attention(q, k, v)
                    relative_rms = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                    self.assertLess(relative_rms, 5e-4)
                    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)

    def test_uniform_weights_and_single_nonzero_key(self):
        q = np.zeros((65, 128), np.float16)
        k = np.zeros_like(q)
        for key in (0, 1, 15, 16, 63, 64):
            v = np.zeros_like(q)
            v[key] = np.arange(128) / 128
            np.testing.assert_allclose(query_attention(q, k, v), dense_attention(q, k, v),
                                       atol=8e-6, rtol=5e-4)

    def test_full_length_keys_and_late_maximum(self):
        random = np.random.default_rng(43)
        q = random.standard_normal((16, 128)).astype(np.float16)
        k = random.standard_normal((4115, 128)).astype(np.float16)
        v = random.standard_normal((4115, 128)).astype(np.float16)
        for late_maximum in (False, True):
            with self.subTest(late_maximum=late_maximum):
                if late_maximum:
                    # Force a large maximum change after 256 online updates,
                    # including the final partial tile and both row parities.
                    k[-16:] = q * 3
                actual = query_attention(q, k, v).astype(np.float64)
                expected = dense_attention(q, k, v)
                self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 5e-4)
                np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)

    def test_32_keys_dense_oracle_and_late_maximum(self):
        random = np.random.default_rng(174)
        for tokens in (17, 31, 32, 33, 65, 4115):
            for scale in (1., 3.):
                with self.subTest(tokens=tokens, scale=scale):
                    q = (random.normal(size=(16, 128)) * scale).astype(np.float16)
                    k = (random.normal(size=(tokens, 128)) * scale).astype(np.float16)
                    v = random.normal(size=(tokens, 128)).astype(np.float16)
                    if tokens == 4115:
                        k[-16:] = q * 3
                    actual = query_attention(q, k, v, key_tile=32).astype(np.float64)
                    expected = dense_attention(q, k, v)
                    self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 5e-4)
                    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)

    def test_48_keys_dense_oracle_and_swizzled_lds(self):
        random = np.random.default_rng(175)
        for tokens in (17, 47, 48, 49, 4115):
            q = random.normal(size=(16, 128)).astype(np.float16)
            k = random.normal(size=(tokens, 128)).astype(np.float16)
            v = random.normal(size=(tokens, 128)).astype(np.float16)
            k[-16:] = q * 3
            actual = query_attention(q, k, v, key_tile=48).astype(np.float64)
            expected = dense_attention(q, k, v)
            self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 5e-4)
            np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
        for rows, columns in ((16, 64), (48, 128), (64, 128), (128, 64)):
            logical = np.arange(rows * columns).reshape(rows, columns)
            physical = np.empty_like(logical)
            for row in range(rows):
                for column in range(0, columns, 8):
                    address = column ^ ((row % 8) * 8)
                    physical[row, address:address + 8] = logical[row, column:column + 8]
            for row in range(rows):
                for column in range(0, columns, 8):
                    address = column ^ ((row % 8) * 8)
                    np.testing.assert_array_equal(physical[row, address:address + 8],
                                                  logical[row, column:column + 8])
            # Eight lanes each read four dwords: every LDS bank appears once.
            for column in range(0, columns, 8):
                banks = [(row * columns // 2 + (column ^ (row * 8)) // 2 + word) % 32
                         for row in range(8) for word in range(4)]
                self.assertEqual(sorted(banks), list(range(32)))

    def test_64_keys_dense_oracle_and_late_maximum(self):
        random = np.random.default_rng(176)
        for tokens in (17, 63, 64, 65, 129, 4115):
            for scale in (1., 3.):
                with self.subTest(tokens=tokens, scale=scale):
                    q = (random.normal(size=(16, 128)) * scale).astype(np.float16)
                    k = (random.normal(size=(tokens, 128)) * scale).astype(np.float16)
                    v = random.normal(size=(tokens, 128)).astype(np.float16)
                    k[-16:] = q * 3
                    actual = query_attention(q, k, v, key_tile=64).astype(np.float64)
                    expected = dense_attention(q, k, v)
                    self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 5e-4)
                    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)

    def test_benchmark_oracle_gqa_heads_and_padded_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            tokens, capacity = 129, 192
            count = prepare_inputs(path, tokens, capacity, 1., 91)
            q = np.fromfile(path / "q.bin", np.float16).reshape(capacity, 48, 128)
            k = np.fromfile(path / "k.bin", np.float16).reshape(capacity, 12, 128)
            v = np.fromfile(path / "v.bin", np.float16).reshape(capacity, 12, 128)
            rows = np.fromfile(path / "rows.bin", np.uint32)
            want = np.fromfile(path / "want.bin", np.float32).reshape(count, 48, 128)
            self.assertIn(tokens - 1, rows)
            for array in (q, k, v):
                self.assertTrue(np.all(array[tokens:] == 0))
            for head in (0, 3, 4, 47):
                expected = dense_attention(q[:tokens, head], k[:tokens, head // 4],
                                           v[:tokens, head // 4])
                np.testing.assert_allclose(want[:, head], expected[rows], atol=3e-8, rtol=1e-7)

    def test_prepare_only_invalidates_old_timing_and_never_launches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiments").mkdir()
            (root / "experiments/attention_query_f16.loom").write_text("candidate source")
            (root / "build").mkdir()
            (root / "build/attention-bench").write_bytes(b"host runner")
            output = root / "output"
            run = output / "attention_query_f16-16-s1.0-seed917"
            run.mkdir(parents=True)
            (run / "result.json").write_text('{"paired_speedup":{"median":2.5}}')
            argv = ["bench_attention.py", "--prepare-only", "--tokens", "16", "--output", str(output)]
            with mock.patch.object(bench_attention, "ROOT", root), \
                 mock.patch.object(bench_attention, "compile_kernel",
                                   side_effect=lambda source, symbol, config, output:
                                   output.write_bytes(symbol.encode())), \
                 mock.patch.object(bench_attention.subprocess, "check_output", return_value=b"baseline source"), \
                 mock.patch.object(bench_attention.subprocess, "run", side_effect=AssertionError("GPU runner launched")), \
                 mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                bench_attention.main()
            self.assertFalse((run / "result.json").exists())
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["oracle_rows"], 16)
            self.assertNotEqual(manifest["baseline_sha256"], manifest["candidate_sha256"])
            self.assertNotEqual(manifest["binary_sha256"]["baseline.hsaco"],
                                manifest["binary_sha256"]["candidate.hsaco"])
            self.assertEqual(len(manifest["runner_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
