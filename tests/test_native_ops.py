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
    for m, n, k, batches, trans, bias in [
        (19, 3, 144, 1, True, False),
        (129, 97, 37, 3, True, False),
        (7, 257, 256, 1, True, True),
        (65, 33, 71, 2, False, False),
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
                    (name + "_bias").encode(),
                    json.dumps(cfg).encode(),
                    buf,
                    len(pack),
                    (n + 63) // 64,
                    batches * ((m + 63) // 64),
                    256,
                )
                y = np.empty((batches, m, n), np.uint16)
                lib.test_copy(y.ctypes.data, pointers[2], y.nbytes)
            finally:
                for ptr in pointers:
                    lib.test_free(ptr)
        else:
            y = run(
                name,
                cfg,
                m,
                [a, b],
                (batches, m, n),
                scalar=1.0,
                grid=((n + 63) // 64, batches * ((m + 63) // 64)),
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
            [x, w],
            x.shape,
            scalar=1e-6,
            grid=(7, 1),
        )
        np.testing.assert_allclose(f32(got), f32(bf(y)), atol=4e-5, rtol=0.008)
    print("PASS activations and all three normalization modes", flush=True)
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


if __name__ == "__main__":
    main()
