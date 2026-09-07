import ctypes as C, json, struct, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
lib = C.CDLL(str(ROOT / "build/libnative_ops_test.so"))
lib.test_alloc.argtypes = [C.c_size_t]
lib.test_alloc.restype = C.c_void_p
lib.test_free.argtypes = [C.c_void_p]
lib.test_copy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
lib.test_run.argtypes = [
    C.c_char_p,
    C.c_char_p,
    C.c_void_p,
    C.c_size_t,
    C.c_uint,
    C.c_uint,
    C.c_uint,
]


def bf(x):
    x = np.asarray(x, dtype=np.float32).copy()
    u = x.view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def f32(x):
    return (np.asarray(x, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def alloc(x):
    x = np.ascontiguousarray(x)
    p = lib.test_alloc(x.nbytes)
    lib.test_copy(p, x.ctypes.data, x.nbytes)
    return p


def run(
    name,
    cfg,
    count,
    arrays,
    shape,
    dtype=np.uint16,
    scalar=None,
    grid=None,
    threads=256,
):
    pointers = [alloc(x) for x in arrays]
    out = np.zeros(shape, dtype=dtype)
    p = alloc(out)
    pointers.append(p)
    pack = struct.pack("i", count)
    if scalar is not None:
        pack += struct.pack("f", scalar)
    pack += bytes((-len(pack)) % 8) + struct.pack("Q" * len(pointers), *pointers)
    b = C.create_string_buffer(pack)
    try:
        assert not lib.test_run(
            name.encode(),
            json.dumps(cfg).encode(),
            b,
            len(pack),
            *(grid or ((count + 255) // 256, 1)),
            threads,
        ), name
        lib.test_copy(out.ctypes.data, p, out.nbytes)
        return out
    finally:
        for p in pointers:
            lib.test_free(p)


def main():
    rng = np.random.default_rng(42)
    # Non-divisible tiles, multiple M tiles, multiple batches, and N >> M catch
    # launch metadata that silently folds a workgroup index to zero.
    for m, n, k, batches, trans, bias, tile_m, tile_n in [
        (19, 3, 144, 1, True, False, 64, 64),
        (129, 97, 37, 3, True, False, 64, 64),
        (7, 257, 256, 1, True, True, 64, 64),
        (65, 33, 71, 2, False, False, 64, 64),
        (257, 97, 37, 3, True, False, 128, 64),
        (129, 128, 132, 2, True, True, 128, 64),
        (192, 64, 128, 2, True, False, 64, 64),
        (257, 129, 37, 2, True, False, 128, 128),
        (129, 128, 132, 2, True, True, 128, 128),
    ]:
        a = bf(rng.standard_normal((batches, m, k)))
        b = bf(rng.standard_normal((batches, n, k) if trans else (batches, k, n)))
        cfg = dict(
            m=m,
            n=n,
            k=k,
            asize=a.size,
            bsize=b.size,
            csize=batches * m * n,
            astride=m * k,
            bstride=n * k,
        )
        name = "gemm_bf16_bf16_" + ("nt" if trans else "nn")
        suffix = "_tiled" if tile_n == 128 else "_wide" if tile_m == 128 else ""
        want = f32(a) @ (f32(b).transpose(0, 2, 1) if trans else f32(b))
        if bias:
            biasval = bf(rng.standard_normal(n))
            want += f32(biasval)
            # The native bias ABI has the result before the bias pointer.
            pointers = [
                alloc(a),
                alloc(b),
                alloc(np.zeros((batches, m, n), np.uint16)),
                alloc(biasval),
            ]
            pack = struct.pack("ifQQQQ", m, 1.0, *pointers)
            buf = C.create_string_buffer(pack)
            try:
                assert not lib.test_run(
                    (name + "_bias" + suffix).encode(),
                    json.dumps(cfg).encode(),
                    buf,
                    len(pack),
                    (n + tile_n - 1) // tile_n,
                    batches * ((m + tile_m - 1) // tile_m),
                    256,
                )
                y = np.empty((batches, m, n), np.uint16)
                lib.test_copy(y.ctypes.data, pointers[2], y.nbytes)
            finally:
                for ptr in pointers:
                    lib.test_free(ptr)
        else:
            y = run(
                name + suffix,
                cfg,
                m,
                [a, b],
                (batches, m, n),
                scalar=1.0,
                grid=(
                    (n + tile_n - 1) // tile_n,
                    batches * ((m + tile_m - 1) // tile_m),
                ),
            )
        np.testing.assert_allclose(f32(y), f32(bf(want)), atol=0.005, rtol=0.008)
    print(
        "PASS BF16 GEMM: tails, batched heads, tall/wide grids, transposes and fused bias",
        flush=True,
    )
    for name in ["silu", "gelu", "sigmoid", "one"]:
        x = bf(np.linspace(-10, 10, 1009))
        y = f32(run("unary_" + name, {}, x.size, [x], x.shape))
        xx = f32(x)
        expected = {
            "silu": xx / (1 + np.exp(-xx)),
            "sigmoid": 1 / (1 + np.exp(-xx)),
            "one": xx + 1,
            "gelu": 0.5
            * xx
            * (1 + np.tanh(0.7978845608028654 * (xx + 0.044715 * xx**3))),
        }[name]
        np.testing.assert_allclose(y, f32(bf(expected)), atol=4e-5, rtol=0.008)
    for mode in range(3):
        x = bf(rng.standard_normal((7, 513)))
        w = bf(rng.standard_normal(513))
        xx = f32(x)
        ww = f32(w)
        inv = 1 / np.sqrt(
            (xx * xx).sum(-1, keepdims=True) / (1 if mode == 2 else 513)
            + (0 if mode == 2 else 1e-6)
        )
        y = xx * inv
        if mode == 0:
            y *= 1 + ww
        elif mode == 1:
            y = f32(bf(y)) * ww
        else:
            y = f32(bf(f32(bf(y)) * np.sqrt(np.float32(513)))) * ww
        got = run(
            "norm_" + str(mode),
            dict(xsize=x.size, cols=513),
            7,
            [x, ww.astype(np.float32)],
            x.shape,
            scalar=1e-6,
            grid=(7, 1),
        )
        np.testing.assert_allclose(f32(got), f32(bf(y)), atol=4e-5, rtol=0.008)
    # Exact reduction order and bf16 rounding must survive both wave packing
    # and norm/SiLU fusion, including partial waves and zero/subnormal rows.
    for cols in (3, 32, 96, 192, 256, 384, 513, 1024):
        x = bf(rng.standard_normal((17, cols)))
        x[0] = 0
        x[1] = bf(np.full(cols, 1e-20, np.float32))
        weights = rng.standard_normal(cols).astype(np.float32)
        cfg = dict(xsize=x.size, cols=cols)
        original = run("norm_2", cfg, 17, [x, weights], x.shape,
                       scalar=1e-5, grid=(17, 1))
        wave = run("norm_2_wave", cfg, 17, [x, weights], x.shape,
                   scalar=1e-5, grid=(3, 1))
        np.testing.assert_array_equal(wave, original)
        separate = run("unary_silu", {}, x.size, [original], x.shape)
        fused = run("norm_2_wave_silu", cfg, 17, [x, weights], x.shape,
                    scalar=1e-5, grid=(3, 1))
        np.testing.assert_array_equal(fused, separate)
    print("PASS activations, normalization modes, and exact wave norm/SiLU fusion", flush=True)
    x = bf(np.arange(5 * 7 * 3).reshape(5, 7, 3))
    padded = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    want = np.array(
        [
            padded[y : y + 3, z : z + 3].transpose(2, 0, 1).reshape(-1)
            for y in range(5)
            for z in range(7)
        ]
    )
    got = run(
        "im2col",
        dict(xsize=x.size, channels=3, width=7, height=5, kernel=3),
        want.size,
        [x],
        want.shape,
    )
    np.testing.assert_array_equal(got, want)
    for h, width, channels in ((1, 1, 32), (3, 7, 96), (8, 5, 192),
                                (7, 9, 384), (2, 3, 1024)):
        source = bf(rng.standard_normal((h, width, channels)))
        padded_source = np.pad(source, ((1, 1), (1, 1), (0, 0)))
        expected = np.array([
            padded_source[y:y + 3, z:z + 3].transpose(2, 0, 1).reshape(-1)
            for y in range(h) for z in range(width)])
        cfg = dict(xsize=source.size, channels=channels, width=width, height=h, kernel=3)
        patches = run("im2col_coalesced", cfg, expected.size, [source],
                      expected.shape, grid=(h * width, 1))
        np.testing.assert_array_equal(patches, expected)
    got = run(
        "upsample",
        dict(xsize=x.size, channels=3, width=7),
        x.size * 4,
        [x],
        (10, 14, 3),
    )
    np.testing.assert_array_equal(got, x.repeat(2, 0).repeat(2, 1))
    x = bf(rng.standard_normal((19, 31)))
    ids = np.array([18, 0, 7], np.int32)
    got = run("embedding", dict(wsize=x.size, cols=31, rows=3), 93, [x, ids], (3, 31))
    np.testing.assert_array_equal(got, x[ids])
    print(
        "PASS convolution padding, nearest upsampling and embedding indexing",
        flush=True,
    )
    for causal in [False, True]:
        x = rng.standard_normal((3, 65, 65)).astype(np.float32)
        masked = x.copy()
        if causal:
            masked[:, np.triu_indices(65, 1)[0], np.triu_indices(65, 1)[1]] = -np.inf
        exp = np.exp(masked - masked.max(-1, keepdims=True))
        want = bf(exp / exp.sum(-1, keepdims=True))
        got = run(
            "softmax_causal" if causal else "softmax",
            dict(xsize=x.size, tokens=65),
            195,
            [x],
            x.shape,
            grid=(195, 1),
        )
        np.testing.assert_allclose(f32(got), f32(want), atol=1e-6, rtol=0.008)
        if causal:
            assert not np.any(
                got[:, np.triu_indices(65, 1)[0], np.triu_indices(65, 1)[1]]
            )
    print("PASS batched softmax and exact causal masking", flush=True)
    # The scheduler ops: Euler's bf16 delta/product rounding and Krea's guidance
    # combine (cond + g * (cond - uncond), rounded per operation as diffusers does).
    n = 1000
    sample, velocity, uncond = (bf(rng.standard_normal(n) * s) for s in (1.0, 0.5, 0.5))
    def launch(name, scalar, first, second):
        pointers = [alloc(first), alloc(second)]
        pack = struct.pack("i", n) + struct.pack("f", scalar)
        pack += bytes((-len(pack)) % 8) + struct.pack("QQ", *pointers)
        b = C.create_string_buffer(pack)
        try:
            assert not lib.test_run(name.encode(), b"{}", b, len(pack), (n + 255) // 256, 1, 256), name
            out = np.zeros(n, dtype=np.uint16)
            lib.test_copy(out.ctypes.data, pointers[0], out.nbytes)
            return out
        finally:
            for p in pointers:
                lib.test_free(p)
    dt = -0.1234
    expected = bf(f32(sample) + f32(bf(f32(bf([dt])) * f32(velocity))))
    assert np.array_equal(launch("euler", dt, sample, velocity), expected), "euler"
    scale = 3.5
    difference = f32(bf(f32(velocity) - f32(uncond)))
    expected = bf(f32(velocity) + f32(bf(scale * difference)))
    assert np.array_equal(launch("guidance", scale, velocity, uncond), expected), "guidance"
    print("PASS Euler step and guidance combine with bf16 rounding", flush=True)


if __name__ == "__main__":
    main()
