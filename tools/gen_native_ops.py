"""Generate the auxiliary Loom kernels embedded in the native library.

Build-time Python only. All tensor arithmetic executes as Loom GPU code.
"""

from pathlib import Path
import os
import subprocess

ROOT = Path(__file__).resolve().parent.parent


class Kernel:
    def __init__(self, name, buffers, configs=(), threads=256, scalars=()):
        self.name, self.buffers, self.configs, self.threads = (
            name,
            buffers,
            configs,
            threads,
        )
        self.scalars = scalars
        self.lines = []
        self.serial = 0
        self.constants = {}
        self.constlines = []
        self.emit("%zero_offset = index.constant 0 : offset")
        self.emit(
            "%count_b = index.assume %count [range(%count, 1, 1073741824)] : index"
        )
        self.emit("%lane = kernel.workitem.id<x> : index")
        self.emit("%group = kernel.workgroup.id<x> : index")
        self.emit("%group_y = kernel.workgroup.id<y> : index")
        for c in configs:
            self.emit(f"%{c} = config.get @krea2.{name}.{c} : index")
        self.i = self.op(
            "index.add", self.op("index.mul", "%group", self.c(threads)), "%lane"
        )
        for b, t, s in buffers:
            self.emit(f"%{b}_global = buffer.assume.memory_space<global> %{b} : buffer")
            self.emit(
                f"%{b}_view = buffer.view %{b}_global[%zero_offset] : buffer -> view<[{s}]x{t}>"
            )

    def emit(self, s):
        self.lines.append(s)

    def var(self):
        self.serial += 1
        return f"%v{self.serial}"

    def c(self, n, t="index"):
        key = (n, t)
        if key not in self.constants:
            v = self.var()
            self.constants[key] = v
            self.constlines.append(
                f"{v} = {'index' if t == 'index' else 'scalar'}.constant {n} : {t}"
            )
        return self.constants[key]

    def op(self, op, *args, t="index"):
        v = self.var()
        self.emit(f"{v} = {op} " + ", ".join(args) + f" : {t}")
        return v

    def add(self, a, b):
        return self.op("index.add", a, b)

    def mul(self, a, b):
        return self.op("index.mul", a, b)

    def div(self, a, b):
        return self.op("index.div", a, b)

    def rem(self, a, b):
        return self.op("index.rem", a, b)

    def cmp(self, a, b, cmp="ult"):
        v = self.var()
        self.emit(f"{v} = index.cmp {cmp}, {a}, {b} : index")
        return v

    def choose(self, c, a, b, t="index"):
        v = self.var()
        self.emit(f"{v} = scf.select {c}, {a}, {b} : {t}")
        return v

    def cast(self, a, src, dst):
        if src == dst:
            return a
        v = self.var()
        if src == "index":
            op = "index.cast"
        elif src == "i32" and dst == "index":
            op = "index.cast"
        elif src == "i32":
            op = "scalar.sitofp"
        elif dst == "i32":
            op = "scalar.fptosi"
        elif dst == "f32":
            op = "scalar.extf"
        else:
            op = "scalar.fptrunc"
        self.emit(f"{v} = {op} {a} : {src} to {dst}")
        return v

    def float(self, a):
        return self.cast(self.cast(a, "index", "i32"), "i32", "f32")

    def rnd(self, a, t="bf16"):
        return self.cast(self.cast(a, "f32", t), t, "f32")

    def load(self, b, i, wide=True):
        _, t, s = next(x for x in self.buffers if x[0] == b)
        ii = self.var()
        self.emit(
            f"{ii} = index.assume {i} [range({i}, 0, 1073741824), lt({i}, {s})] : index"
        )
        v = self.var()
        self.emit(f"{v} = view.load %{b}_view[{ii}] : view<[{s}]x{t}> -> {t}")
        return self.cast(v, t, "f32") if wide and t in ("bf16", "f16") else v

    def store(self, b, i, v, src="f32"):
        _, t, s = next(x for x in self.buffers if x[0] == b)
        v = self.cast(v, src, t)
        ii = self.var()
        self.emit(
            f"{ii} = index.assume {i} [range({i}, 0, 1073741824), lt({i}, {s})] : index"
        )
        self.emit(f"view.store {v}, %{b}_view[{ii}] : {t}, view<[{s}]x{t}>")

    def math(self, op, *a):
        if op in ("expf", "powf", "sinf", "cosf", "tanhf"):
            op += "<afn>"
        return self.op("scalar." + op, *a, t="f32")

    def begin(self, c):
        self.emit(f"scf.if {c} {{")

    def end(self):
        self.emit("}")

    def guard(self):
        self.begin(self.cmp(self.i, "%count_b"))

    def loop(self, start, end, step, initial, body, t="f32"):
        i, acc, out = self.var(), self.var(), self.var()
        self.emit(
            f"{out} = scf.for {i} = [{start} to {end} step {step}]({acc} = {initial} : {t}) -> ({t}) {{"
        )
        val = body(i, acc)
        self.emit(f"scf.yield {val} : {t}")
        self.end()
        return out

    def text(self):
        n = self.name
        cfg = "\n".join(
            f"config.decl @krea2.{n}.{c} : %value: index where [range(%value, 1, 1073741824)]"
            for c in [*self.configs, "grid_x", "grid_y"]
        )
        args = ", ".join(
            [f"%{n}: {t}" for n, t in self.scalars]
            + [f"%{b}: buffer" for b, _, _ in self.buffers]
        )
        return (
            f"""// Generated by tools/gen_native_ops.py. gfx1151 auxiliary operation.
amdgpu.target<gfx11-generic> @krea2_{n}_target {{subgroup_size = 32}}
{cfg}
kernel.def target(@krea2_{n}_target) export("krea2_{n}") @krea2_{n}(%count: index) {{
  %one = index.constant 1 : index
  %threads = index.constant {self.threads} : index
  %grid_x = config.get @krea2.{n}.grid_x : index
  %grid_y = config.get @krea2.{n}.grid_y : index
  kernel.launch.config workgroups(%grid_x, %grid_y, %one) workgroup_size(%threads, %one, %one) : index
}} launch(%count: index, {args}) {{
"""
            + "\n".join("  " + s for s in self.constlines + self.lines)
            + "\n  kernel.return\n}\n"
        )


sources = {}


def save(k):
    sources[k.name] = k.text()


def point(name, buffers, configs, body, scalars=()):
    k = Kernel(name, buffers, configs, scalars=scalars)
    k.guard()
    body(k)
    k.end()
    save(k)


def unary(k, op):
    x = k.load("x", k.i)
    one = k.c("1.0", "f32")
    if op == "one":
        y = k.math("addf", one, x)
    else:
        if op == "gelu":
            cubic = k.math("mulf", k.math("mulf", x, x), x)
            a = k.math(
                "mulf",
                k.c("0.7978845608028654", "f32"),
                k.math("addf", x, k.math("mulf", k.c("0.044715", "f32"), cubic)),
            )
            y = k.math(
                "mulf",
                k.math("mulf", k.c("0.5", "f32"), x),
                k.math("addf", one, k.math("tanhf", a)),
            )
        else:
            den = k.math("addf", one, k.math("expf", k.math("negf", x)))
            y = k.math("divf", x if op == "silu" else one, den)
    k.store("y", k.i, y)


for name in ["silu", "gelu", "sigmoid", "one"]:
    point(
        "unary_" + name,
        [("x", "bf16", "%count_b"), ("y", "bf16", "%count_b")],
        [],
        lambda k, n=name: unary(k, n),
    )
for name, op in [("add", "addf"), ("mul", "mulf")]:
    point(
        "binary_" + name,
        [("x", "bf16", "%count_b"), ("y", "bf16", "%yn"), ("z", "bf16", "%count_b")],
        ["yn"],
        lambda k, o=op: k.store(
            "z", k.i, k.math(o, k.load("x", k.i), k.load("y", k.rem(k.i, "%yn")))
        ),
    )
for a, b in [("bf16", "f16"), ("f16", "bf16")]:
    point(
        "cast_" + a + "_" + b,
        [("x", a, "%count_b"), ("y", b, "%count_b")],
        [],
        lambda k: k.store("y", k.i, k.load("x", k.i)),
    )


def euler(k):
    dt = k.rnd("%delta")
    product = k.rnd(k.math("mulf", dt, k.load("velocity", k.i)))
    k.store("sample", k.i, k.math("addf", k.load("sample", k.i), product))


point(
    "euler",
    [("sample", "bf16", "%count_b"), ("velocity", "bf16", "%count_b")],
    [],
    euler,
    scalars=[("delta", "f32")],
)


def guidance(k):
    # Krea's classifier-free guidance, in place on the conditional velocity with
    # diffusers' bf16 rounding at each of its three tensor operations:
    #   cond = cond + scale * (cond - uncond)
    cond = k.load("cond", k.i)
    difference = k.rnd(k.math("subf", cond, k.load("uncond", k.i)))
    scaled = k.rnd(k.math("mulf", "%scale", difference))
    k.store("cond", k.i, k.math("addf", cond, scaled))


point(
    "guidance",
    [("cond", "bf16", "%count_b"), ("uncond", "bf16", "%count_b")],
    [],
    guidance,
    scalars=[("scale", "f32")],
)


# Copy/index transforms. Their actual input/output extents are supplied as configs.
def columns(k):
    j = k.add(
        k.mul(k.div(k.i, "%cols"), "%width"),
        k.add(k.rem(k.i, "%cols"), k.op("index.sub", "%start1", k.c(1))),
    )
    k.store("y", k.i, k.load("x", j))


point(
    "columns",
    [("x", "bf16", "%xsize"), ("y", "bf16", "%count_b")],
    ["xsize", "cols", "width", "start1"],
    columns,
)


def embedding(k):
    row = k.div(k.i, "%cols")
    ident = k.cast(k.load("ids", row, False), "i32", "index")
    k.store("y", k.i, k.load("w", k.add(k.mul(ident, "%cols"), k.rem(k.i, "%cols"))))


point(
    "embedding",
    [("w", "bf16", "%wsize"), ("ids", "i32", "%rows"), ("y", "bf16", "%count_b")],
    ["wsize", "rows", "cols"],
    embedding,
)


def tap(k):
    row = k.div(k.i, k.c(2560))
    col = k.rem(k.i, k.c(2560))
    src = k.add(k.mul(k.add(row, k.c(34)), k.c(2560)), col)
    dst = k.add(
        k.mul(
            k.add(k.mul(row, k.c(12)), k.op("index.sub", "%tap1", k.c(1))), k.c(2560)
        ),
        col,
    )
    k.store("y", dst, k.load("x", src))


point(
    "tap",
    [("x", "bf16", "%xsize"), ("y", "bf16", "%ysize")],
    ["xsize", "ysize", "tap1"],
    tap,
)


def fuse(k):
    base = k.add(k.mul(k.div(k.i, k.c(2560)), k.c(12 * 2560)), k.rem(k.i, k.c(2560)))
    out = k.loop(
        k.c(0),
        k.c(12),
        k.c(1),
        k.c("0.0", "f32"),
        lambda j, a: k.math(
            "addf",
            a,
            k.math(
                "mulf", k.load("x", k.add(base, k.mul(j, k.c(2560)))), k.load("w", j)
            ),
        ),
    )
    k.store("y", k.i, out)


point(
    "fuse",
    [("x", "bf16", "%xsize"), ("w", "bf16", "%wsize"), ("y", "bf16", "%count_b")],
    ["xsize", "wsize"],
    fuse,
)
point(
    "modulation",
    [("x", "bf16", "%xsize"), ("tables", "bf16", "%count_b"), ("y", "f32", "%count_b")],
    ["xsize"],
    lambda k: k.store(
        "y",
        k.i,
        k.rnd(k.math("addf", k.load("x", k.rem(k.i, "%xsize")), k.load("tables", k.i))),
    ),
)


def upsample(k):
    p = k.div(k.i, "%channels")
    ww = k.mul("%width", k.c(2))
    pix = k.add(
        k.mul(k.div(k.div(p, ww), k.c(2)), "%width"), k.div(k.rem(p, ww), k.c(2))
    )
    k.store(
        "y", k.i, k.load("x", k.add(k.mul(pix, "%channels"), k.rem(k.i, "%channels")))
    )


point(
    "upsample",
    [("x", "bf16", "%xsize"), ("y", "bf16", "%count_b")],
    ["xsize", "channels", "width"],
    upsample,
)


def im2col(k):
    kk = k.mul("%kernel", "%kernel")
    stride = k.mul("%channels", kk)
    p = k.div(k.i, stride)
    q = k.rem(k.i, stride)
    yy = k.op(
        "index.sub",
        k.add(k.div(p, "%width"), k.rem(k.div(q, "%kernel"), "%kernel")),
        k.div("%kernel", k.c(2)),
    )
    xx = k.op(
        "index.sub",
        k.add(k.rem(p, "%width"), k.rem(q, "%kernel")),
        k.div("%kernel", k.c(2)),
    )
    valid = k.op("scalar.andi", k.cmp(yy, "%height"), k.cmp(xx, "%width"), t="i1")
    k.begin(valid)
    y0, x0 = yy, xx
    yy, xx = k.var(), k.var()
    k.emit(
        f"{yy} = index.assume {y0} [range({y0}, 0, 1048576), lt({y0}, %height)] : index"
    )
    k.emit(
        f"{xx} = index.assume {x0} [range({x0}, 0, 1048576), lt({x0}, %width)] : index"
    )
    pos = k.add(k.mul(k.add(k.mul(yy, "%width"), xx), "%channels"), k.div(q, kk))
    k.store("y", k.i, k.load("x", pos))
    k.emit("} else {")
    k.store("y", k.i, k.c("0.0", "f32"))
    k.end()


point(
    "im2col",
    [("x", "bf16", "%xsize"), ("y", "bf16", "%count_b")],
    ["xsize", "channels", "width", "height", "kernel"],
    im2col,
)


# Sequential per-lane sums followed by the exact 256-lane tree used by the oracle.
def norm(mode):
    k = Kernel(
        "norm_" + str(mode),
        [("x", "bf16", "%xsize"), ("w", "f32", "%cols"), ("y", "bf16", "%xsize")],
        ["xsize", "cols"],
        scalars=[("eps", "f32")],
    )
    base = k.mul("%group", "%cols")
    total = k.loop(
        "%lane",
        "%cols",
        k.c(256),
        k.c("0.0", "f32"),
        lambda j, a: k.math(
            "addf",
            a,
            k.math("mulf", k.load("x", k.add(base, j)), k.load("x", k.add(base, j))),
        ),
    )
    k.emit("%lds_bytes = index.constant 1024 : offset")
    k.emit("%lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer")
    k.emit("%sums = buffer.view %lds[%zero_offset] : buffer -> view<256xf32>")
    k.emit(f"view.store {total}, %sums[%lane] : f32, view<256xf32>")
    for d in [128, 64, 32, 16, 8, 4, 2, 1]:
        k.emit("kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        k.begin(k.cmp("%lane", k.c(d)))
        peer = k.add("%lane", k.c(d))
        v = k.var()
        w = k.var()
        k.emit(f"{v} = view.load %sums[%lane] : view<256xf32> -> f32")
        k.emit(f"{w} = view.load %sums[{peer}] : view<256xf32> -> f32")
        s = k.math("addf", v, w)
        k.emit(f"view.store {s}, %sums[%lane] : f32, view<256xf32>")
        k.end()
    k.emit("kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    s = k.var()
    k.emit(f"{s} = view.load %sums[{k.c(0)}] : view<256xf32> -> f32")
    cf = k.float("%cols")
    if mode == 2:
        inv = k.math(
            "divf",
            k.c("1.0", "f32"),
            k.math("maxnumf", k.math("sqrtf", s), k.c("1e-12", "f32")),
        )
    else:
        inv = k.math("rsqrtf", k.math("addf", k.math("divf", s, cf), "%eps"))
    j = k.var()
    k.emit(f"scf.for {j} = [%lane to %cols step {k.c(256)}] {{")
    x = k.math("mulf", k.load("x", k.add(base, j)), inv)
    w = k.load("w", j)
    if mode == 0:
        y = k.math("mulf", x, k.math("addf", k.c("1.0", "f32"), w))
    elif mode == 1:
        y = k.math("mulf", k.rnd(x), w)
    else:
        y = k.math("mulf", k.rnd(k.math("mulf", k.rnd(x), k.math("sqrtf", cf))), w)
    k.store("y", k.add(base, j), y)
    k.end()
    save(k)


for mode in range(3):
    norm(mode)


def rope(k):
    d = k.rem(k.i, "%dim")
    half = k.div("%dim", k.c(2))
    lower = k.cmp(d, half)
    other = k.choose(lower, k.add(d, half), k.op("index.sub", d, half))
    position = k.div(k.div(k.i, "%dim"), "%heads")
    exponent = k.math(
        "divf",
        k.math("mulf", k.c("-2.0", "f32"), k.float(k.rem(d, half))),
        k.float("%dim"),
    )
    angle = k.math("mulf", k.float(position), k.math("powf", "%theta", exponent))
    co = k.rnd(k.math("cosf", angle))
    si = k.rnd(k.math("sinf", angle))
    sign = k.choose(lower, k.c("-1.0", "f32"), k.c("1.0", "f32"), t="f32")
    left = k.rnd(k.math("mulf", k.load("x", k.i), co))
    right = k.rnd(
        k.math(
            "mulf",
            k.math("mulf", sign, k.load("x", k.add(k.op("index.sub", k.i, d), other))),
            si,
        )
    )
    k.store("y", k.i, k.math("addf", left, right))


point(
    "rope",
    [("x", "bf16", "%count_b"), ("y", "bf16", "%count_b")],
    ["dim", "heads"],
    rope,
    scalars=[("theta", "f32")],
)


def head_pack(k, unpack):
    d = k.rem(k.i, "%dim")
    s = k.rem(k.div(k.i, "%dim"), "%tokens")
    h = k.rem(k.div(k.div(k.i, "%dim"), "%tokens"), "%heads")
    b = k.div(k.div(k.div(k.i, "%dim"), "%tokens"), "%heads")
    j = k.add(
        k.mul(
            k.add(
                k.mul(k.add(k.mul(b, "%tokens"), s), "%kv"),
                k.div(h, k.div("%heads", "%kv")),
            ),
            "%dim",
        ),
        d,
    )
    if unpack:
        k.store("y", j, k.load("x", k.i))
    else:
        k.store("y", k.i, k.load("x", j))


for unpack in [False, True]:
    point(
        "head_unpack" if unpack else "head_pack",
        [("x", "bf16", "%xsize"), ("y", "bf16", "%ysize")],
        ["dim", "tokens", "heads", "kv", "xsize", "ysize"],
        lambda k, u=unpack: head_pack(k, u),
    )


def softmax(causal):
    k = Kernel(
        "softmax_causal" if causal else "softmax",
        [("x", "f32", "%xsize"), ("y", "bf16", "%xsize")],
        ["xsize", "tokens"],
    )
    base = k.mul("%group", "%tokens")
    query = k.rem("%group", "%tokens")
    end = k.add(query, k.c(1)) if causal else "%tokens"

    def score(j):
        return k.load("x", k.add(base, j))

    maximum = k.loop(
        "%lane",
        end,
        k.c(256),
        k.c("-3.402823466e+38", "f32"),
        lambda j, a: k.math("maxnumf", a, score(j)),
    )
    k.emit("%lds_bytes = index.constant 2048 : offset")
    k.emit("%lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer")
    k.emit("%sum_offset = index.constant 1024 : offset")
    k.emit("%max_buf = buffer.view %lds[%zero_offset] : buffer -> view<256xf32>")
    k.emit("%sum_buf = buffer.view %lds[%sum_offset] : buffer -> view<256xf32>")

    def reduce(value, op, buf):
        k.emit(f"view.store {value}, {buf}[%lane] : f32, view<256xf32>")
        for d in [128, 64, 32, 16, 8, 4, 2, 1]:
            k.emit("kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            k.begin(k.cmp("%lane", k.c(d)))
            peer = k.add("%lane", k.c(d))
            a = k.var()
            b = k.var()
            k.emit(f"{a} = view.load {buf}[%lane] : view<256xf32> -> f32")
            k.emit(f"{b} = view.load {buf}[{peer}] : view<256xf32> -> f32")
            v = k.math(op, a, b)
            k.emit(f"view.store {v}, {buf}[%lane] : f32, view<256xf32>")
            k.end()
        k.emit("kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        v = k.var()
        k.emit(f"{v} = view.load {buf}[{k.c(0)}] : view<256xf32> -> f32")
        return v

    # Keep the maximum alive in its own LDS region. The gfx1151 lowering can
    # emit ds_load; s_barrier without waiting for that read to finish. Reusing
    # the same region lets a faster wave overwrite the maximum with a partial
    # sum while another wave's broadcast read is still outstanding. Separate
    # regions also eliminate the two post-broadcast barriers.
    mx = reduce(maximum, "maxnumf", "%max_buf")
    su = k.loop(
        "%lane",
        end,
        k.c(256),
        k.c("0.0", "f32"),
        lambda j, a: k.math("addf", a, k.math("expf", k.math("subf", score(j), mx))),
    )
    total = reduce(su, "addf", "%sum_buf")
    j = k.var()
    k.emit(f"scf.for {j} = [%lane to %tokens step {k.c(256)}] {{")
    k.begin(k.cmp(j, end))
    v = k.math("divf", k.math("expf", k.math("subf", score(j), mx)), total)
    k.store("y", k.add(base, j), v)
    k.emit("} else {")
    k.store("y", k.add(base, j), k.c("0.0", "f32"))
    k.end()
    k.end()
    save(k)


softmax(False)
softmax(True)


# 64x64, 128x64 and 128x128 WMMA tiles, using a 4x2 wave layout.
# Larger row tiles reuse B fragments; larger column tiles halve A traffic
# in the VAE's long reductions. All share the same accumulation order.
# Scalar predication at the edges handles arbitrary M/N/K, including RGB's N=3.
def gemm(dtype="bf16", output="bf16", transpose=True, bias=False, tile_m=64, tile_n=64):
    name = (
        "gemm_"
        + dtype
        + "_"
        + output
        + ("_nt" if transpose else "_nn")
        + ("_bias" if bias else "")
        + ("_tiled" if tile_n == 128 else "_wide" if tile_m == 128 else "")
    )
    bufs = [("a", dtype, "%asize"), ("b", dtype, "%bsize"), ("out", output, "%csize")]
    if bias:
        bufs.append(("bias", dtype, "%n"))
    k = Kernel(
        name,
        bufs,
        ["m", "n", "k", "asize", "bsize", "csize", "astride", "bstride"],
        scalars=[("alpha", "f32")],
    )
    cm = [k.c(i) for i in [0, 1, 2, 4, 8, 16, 32, 40, 64, 256]]
    c0, c1, c2, c4, c8, c16, c32, c40, c64, c256 = cm
    wave = k.var()
    lane = k.var()
    k.emit(f"{wave} = kernel.subgroup.id : index")
    k.emit(f"{lane} = kernel.subgroup.lane.id : index")
    wr = k.mul(k.div(wave, c2), k.c(tile_m // 4))
    wc = k.mul(k.rem(wave, c2), k.c(tile_n // 2))
    mt = k.div(k.add("%m", k.c(tile_m - 1)), k.c(tile_m))
    batch = k.div("%group_y", mt)
    bm = k.mul(k.rem("%group_y", mt), k.c(tile_m))
    bn = k.mul("%group", k.c(tile_n))
    ao = k.mul(batch, "%astride")
    bo = k.mul(batch, "%bstride")
    co = k.mul(k.mul(batch, "%m"), "%n")
    # Once the K loop's final workgroup barrier completes, its operand tiles
    # are dead. Reuse that LDS for the eight result tiles (8192 bytes).
    k.emit(f"%stage_bytes = index.constant {(tile_m + tile_n) * 80} : offset")
    k.emit(f"%b_offset = index.constant {tile_m * 80} : offset")
    k.emit("%wave_bytes = index.constant 1024 : offset")
    k.emit("%stage_buf = buffer.alloca<workgroup> align(16) %stage_bytes : buffer")
    for s in ["aa", "bb"]:
        offset = "%zero_offset" if s == "aa" else "%b_offset"
        k.emit(
            f"%{s} = buffer.view %stage_buf[{offset}] : buffer -> view<{tile_m if s == 'aa' else tile_n}x40x{dtype}>"
        )
    k.emit(f"%rhs_layout = encoding.layout.strided [1, {c40}] : encoding<layout>")
    k.emit(
        f"%rhs = buffer.view %stage_buf[%b_offset] : buffer -> view<32x{tile_n}x{dtype}, %rhs_layout>"
    )
    ro = k.var()
    k.emit(f"{ro} = index.scale {wave}, %wave_bytes : index, offset -> offset")
    k.emit(f"%result = buffer.view %stage_buf[{ro}] : buffer -> view<16x16xf32>")
    k.emit("%zero_vec = vector.constant 0.0 : vector<8xf32>")
    k.emit(
        f"%init = vector.fragment<init> %zero_vec shape [{c16}, {c16}] : vector<8xf32>"
    )
    loadrow = k.div("%lane", c8)
    loadcol = k.mul(k.rem("%lane", c8), c4)
    kend = k.mul(k.div(k.add("%k", k.c(31)), c32), c32)
    fm, fn = tile_m // 64, tile_n // 32
    accumulators = [k.var() for _ in range(fm * fn)]
    results = [k.var() for _ in accumulators]
    kb = k.var()
    types = ", ".join(["vector<8xf32>"] * len(accumulators))
    initials = ", ".join(f"{a} = %init : vector<8xf32>" for a in accumulators)
    k.emit(
        f"{', '.join(results)} = scf.for {kb} = [{c0} to {kend} step {c32}]({initials}) -> ({types}) {{"
    )
    for h in range(max(tile_m, tile_n) // 32):
        row = k.add(loadrow, k.c(h * 32))
        ar = k.add(bm, row)
        br = k.add(bn, row)
        # A contiguous four-element packet replaces four separately predicated
        # global loads and LDS stores. K alignment is known at compilation;
        # retain scalar tails for the small irregular attention matrices.
        if transpose:
            aligned = k.cmp(k.rem("%k", c4), c0, "eq")
            k.begin(aligned)
            col = k.add(kb, loadcol)
            for inp, rr, limit, off, tile in [
                ("a", ar, "%m", ao, "aa"),
                ("b", br, "%n", bo, "bb"),
            ]:
                if h >= (tile_m if inp == "a" else tile_n) // 32:
                    continue
                valid = k.op("scalar.andi", k.cmp(rr, limit), k.cmp(col, "%k"), t="i1")
                value = k.var()
                k.emit(f"{value} = scf.if {valid} -> (vector<4x{dtype}>) {{")
                idx = k.add(off, k.add(k.mul(rr, "%k"), col))
                bounded = k.var()
                size = "%asize" if inp == "a" else "%bsize"
                end = k.op("index.sub", size, c4)
                k.emit(
                    f"{bounded} = index.assume {idx} [range({idx}, 0, 1073741823), le({idx}, {end}), mul({idx}, 4)] : index"
                )
                val = k.var()
                k.emit(
                    f"{val} = vector.load %{inp}_view[{bounded}] : view<[{size}]x{dtype}> -> vector<4x{dtype}>"
                )
                k.emit(f"scf.yield {val} : vector<4x{dtype}>")
                k.emit("} else {")
                zero = k.var()
                k.emit(f"{zero} = vector.constant 0.0 : vector<4x{dtype}>")
                k.emit(f"scf.yield {zero} : vector<4x{dtype}>")
                k.end()
                k.emit(
                    f"vector.store {value}, %{tile}[{row}, {loadcol}] : vector<4x{dtype}>, view<{tile_m if tile == 'aa' else tile_n}x40x{dtype}>"
                )
            k.emit("} else {")
        for j in range(4):
            lc = k.add(loadcol, k.c(j))
            col = k.add(kb, lc)
            for inp, rr, limit, off, tile in [
                ("a", ar, "%m", ao, "aa"),
                ("b", br, "%n", bo, "bb"),
            ]:
                if h >= (tile_m if inp == "a" else tile_n) // 32:
                    continue
                valid = k.op("scalar.andi", k.cmp(rr, limit), k.cmp(col, "%k"), t="i1")
                value = k.var()
                k.emit(f"{value} = scf.if {valid} -> ({dtype}) {{")
                idx = k.add(
                    off,
                    k.add(k.mul(col, "%n"), rr)
                    if inp == "b" and not transpose
                    else k.add(k.mul(rr, "%k"), col),
                )
                val = k.load(inp, idx, False)
                k.emit(f"scf.yield {val} : {dtype}")
                k.emit("} else {")
                z = k.c("0.0", dtype)
                k.emit(f"scf.yield {z} : {dtype}")
                k.end()
                k.emit(
                    f"view.store {value}, %{tile}[{row}, {lc}] : {dtype}, view<{tile_m if tile == 'aa' else tile_n}x40x{dtype}>"
                )
        if transpose:
            k.end()
    k.emit("kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    values = accumulators[:]
    for h in range(2):
        kh = k.c(h * 16)
        rhs = []
        for j in range(fn):
            v = k.var()
            col = k.add(wc, k.c(j * 16))
            k.emit(
                f"{v} = vector.fragment.load<rhs> %rhs[{kh}, {col}] shape [{c16}, {c16}] : view<32x{tile_n}x{dtype}, %rhs_layout> -> vector<16x{dtype}>"
            )
            rhs.append(v)
        for i in range(fm):
            lhs = k.var()
            row = k.add(wr, k.c(i * 16))
            k.emit(
                f"{lhs} = vector.fragment.load<lhs> %aa[{row}, {kh}] shape [{c16}, {c16}] : view<{tile_m}x40x{dtype}> -> vector<16x{dtype}>"
            )
            for j in range(fn):
                v = k.var()
                k.emit(
                    f"{v} = vector.mma {lhs}, {rhs[j]}, {values[i * fn + j]} : vector<16x{dtype}>, vector<16x{dtype}>, vector<8xf32>"
                )
                values[i * fn + j] = v
    k.emit("kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    k.emit(f"scf.yield {', '.join(values)} : {types}")
    k.end()
    pr = k.div(lane, c4)
    pc = k.mul(k.rem(lane, c4), c4)
    for f, acc in enumerate(results):
        k.emit(
            f"vector.fragment.store<result> {acc}, %result[{c0}, {c0}] shape [{c16}, {c16}] : vector<8xf32>, view<16x16xf32>"
        )
        k.emit("kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)")
        for h in range(2):
            rr = k.add(pr, k.c(h * 8))
            row = k.add(bm, k.add(wr, k.add(k.c((f // fn) * 16), rr)))
            # Aligned output rows can leave LDS in four-element packets too.
            # N=3 RGB and other irregular widths use the scalar edge path.
            aligned = k.cmp(k.rem("%n", c4), c0, "eq")
            k.begin(aligned)
            col = k.add(bn, k.add(wc, k.add(k.c((f % fn) * 16), pc)))
            valid = k.op("scalar.andi", k.cmp(row, "%m"), k.cmp(col, "%n"), t="i1")
            k.begin(valid)
            value, alpha, scaled = [k.var() for _ in range(3)]
            k.emit(
                f"{value} = vector.load %result[{rr}, {pc}] : view<16x16xf32> -> vector<4xf32>"
            )
            k.emit(f"{alpha} = vector.splat %alpha : vector<4xf32>")
            k.emit(f"{scaled} = vector.mulf {value}, {alpha} : vector<4xf32>")
            if bias:
                bc, bv, bf, added = [k.var() for _ in range(4)]
                end = k.op("index.sub", "%n", c4)
                k.emit(
                    f"{bc} = index.assume {col} [range({col}, 0, 1073741823), le({col}, {end}), mul({col}, 4)] : index"
                )
                k.emit(
                    f"{bv} = vector.load %bias_view[{bc}] : view<[%n]x{dtype}> -> vector<4x{dtype}>"
                )
                k.emit(f"{bf} = vector.extf {bv} : vector<4x{dtype}> to vector<4xf32>")
                k.emit(f"{added} = vector.addf {scaled}, {bf} : vector<4xf32>")
                scaled = added
            if output != "f32":
                narrow = k.var()
                k.emit(
                    f"{narrow} = vector.fptrunc {scaled} : vector<4xf32> to vector<4x{output}>"
                )
                scaled = narrow
            idx = k.add(co, k.add(k.mul(row, "%n"), col))
            bound = k.var()
            end = k.op("index.sub", "%csize", c4)
            k.emit(
                f"{bound} = index.assume {idx} [range({idx}, 0, 1073741823), le({idx}, {end}), mul({idx}, 4)] : index"
            )
            k.emit(
                f"vector.store {scaled}, %out_view[{bound}] : vector<4x{output}>, view<[%csize]x{output}>"
            )
            k.end()
            k.emit("} else {")
            for j in range(4):
                cc = k.add(pc, k.c(j))
                col = k.add(bn, k.add(wc, k.add(k.c((f % fn) * 16), cc)))
                valid = k.op("scalar.andi", k.cmp(row, "%m"), k.cmp(col, "%n"), t="i1")
                k.begin(valid)
                v = k.var()
                k.emit(f"{v} = view.load %result[{rr}, {cc}] : view<16x16xf32> -> f32")
                v = k.math("mulf", v, "%alpha")
                if bias:
                    v = k.math("addf", v, k.load("bias", col))
                k.store("out", k.add(co, k.add(k.mul(row, "%n"), col)), v)
                k.end()
            k.end()
        k.emit("kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)")
    save(k)


gemm(tile_m=128, tile_n=128)
gemm(bias=True, tile_m=128, tile_n=128)
gemm(tile_m=128)
gemm(bias=True, tile_m=128)

for kw in [
    dict(),
    dict(bias=True),
    dict(output="f32"),
    dict(transpose=False),
    dict(dtype="f16", output="f32"),
]:
    gemm(**kw)


def means(kind):
    name = "sage_" + kind
    cfg = ["tokens", "heads", "tiles", "xsize", "ysize"]
    bufs = [
        ("x", "f32" if kind == "key_mean" else "f16", "%xsize"),
        ("y", "f32", "%ysize"),
    ]
    if kind == "query_mean":
        bufs.append(("half", "f16", "%ysize"))
    k = Kernel(name, bufs, cfg, threads=128)
    c128 = k.c(128)
    c64 = k.c(64)
    if kind == "key_mean":
        end = "%tiles"
        start = k.c(0)

        def load(j):
            return k.load(
                "x", k.add(k.mul(k.add(k.mul(j, "%heads"), "%group"), c128), "%lane")
            )

        count = k.float("%tokens")
        out = k.add(k.mul("%group", c128), "%lane")
    else:
        tile = "%group_y" if kind == "key_partial" else "%group"
        head = "%group" if kind == "key_partial" else "%group_y"
        start = k.mul(tile, c64)
        end = k.op("index.min", k.add(start, c64), "%tokens")

        def load(j):
            return k.load(
                "x", k.add(k.mul(k.add(k.mul(j, "%heads"), head), c128), "%lane")
            )

        count = k.float(k.op("index.sub", end, start))
        out = k.add(
            k.mul(
                k.add(k.mul(tile, "%heads"), head)
                if kind == "key_partial"
                else k.add(k.mul(head, "%tiles"), tile),
                c128,
            ),
            "%lane",
        )
    total = k.loop(
        start, end, k.c(1), k.c("0.0", "f32"), lambda j, a: k.math("addf", a, load(j))
    )
    if kind != "key_partial":
        total = k.math("divf", total, count)
    k.store("y", out, total)
    if kind == "query_mean":
        k.store("half", out, total)
    save(k)


for kind in ["key_partial", "key_mean", "query_mean"]:
    means(kind)


def quant(query, bits=4):
    # One wave handles one 128-channel row, four channels per lane. Eight rows
    # share a workgroup, with no LDS allocation or workgroup barriers.
    # bits 4: codes in -7..7, four per i16 (64 B per head row); bits 8: -127..127,
    # four per i32 (128 B per head row). Either way 32 words per head row.
    cfg = ["tokens", "heads", "tiles", "capacity", "xsize", "msize", "psize", "ssize"]
    word_type, levels, mask, shift = ("i16", 7, 15, 4) if bits == 4 else ("i32", 127, 255, 8)
    bufs = [
        ("x", "f16", "%xsize"),
        ("mean", "f32", "%msize"),
        ("packed", word_type, "%psize"),
        ("scale", "f32", "%ssize"),
    ]
    if not query:
        bufs.append(("centered", "f16", "%xsize"))
    k = Kernel(("sage_quant_q" if query else "sage_quant_k") + ("" if bits == 4 else "_i8"), bufs, cfg)
    lane = k.rem("%lane", k.c(32))
    row = k.add(k.mul("%group", k.c(8)), k.div("%lane", k.c(32)))
    head = "%group_y"
    k.begin(k.cmp(row, "%tokens" if query else "%capacity"))
    values = []
    maximum = k.c("0.0", "f32")
    for j in range(4):
        channel = k.add(k.mul(lane, k.c(4)), k.c(j))
        valid = k.cmp(row, "%tokens")
        val = k.var()
        k.emit(f"{val} = scf.if {valid} -> (f32) {{")
        xi = k.add(k.mul(k.add(k.mul(row, "%heads"), head), k.c(128)), channel)
        meanrow = k.add(k.mul(head, "%tiles"), k.div(row, k.c(64))) if query else head
        mi = k.add(k.mul(meanrow, k.c(128)), channel)
        value = k.math("subf", k.load("x", xi), k.load("mean", mi))
        k.emit(f"scf.yield {value} : f32")
        k.emit("} else {")
        k.emit(f"scf.yield {k.c('0.0', 'f32')} : f32")
        k.end()
        values.append(val)
        maximum = k.math("maxnumf", maximum, k.math("absf", val))
        if not query:
            at = k.add(k.mul(k.add(k.mul(head, "%capacity"), row), k.c(128)), channel)
            k.store("centered", at, val)
    for delta in [16, 8, 4, 2, 1]:
        other, valid = k.var(), k.var()
        k.emit(
            f"{other}, {valid} = kernel.subgroup.shuffle<xor> {maximum}, {k.c(delta, 'i32')}, {k.c(32, 'i32')} : f32, i32, i32"
        )
        maximum = k.math("maxnumf", maximum, other)
    scale = k.math(
        "divf", k.math("maxnumf", maximum, k.c("1e-30", "f32")), k.c(f"{levels}.0", "f32")
    )
    # Head-major outputs: codes [heads][capacity][64 B], scales [heads][capacity], so a
    # key tile is one contiguous block for the attention kernel's staging loads.
    k.begin(k.cmp(lane, k.c(0), "eq"))
    k.store("scale", k.add(k.mul(head, "%capacity"), row), scale)
    k.end()
    word = k.c(0, "i32")
    for j, value in enumerate(values):
        code = k.math("roundevenf", k.math("divf", value, scale))
        code = k.math(
            "minnumf", k.math("maxnumf", code, k.c(f"-{levels}.0", "f32")), k.c(f"{levels}.0", "f32")
        )
        code = k.op("scalar.andi", k.cast(code, "f32", "i32"), k.c(mask, "i32"), t="i32")
        word = k.op(
            "scalar.ori",
            word,
            k.op("scalar.shli", code, k.c(shift * j, "i32"), t="i32"),
            t="i32",
        )
    at = k.add(k.mul(k.add(k.mul(head, "%capacity"), row), k.c(32)), lane)
    if bits == 4:
        packed = k.var()
        k.emit(f"{packed} = scalar.trunci {word} : i32 to i16")
        k.store("packed", at, packed, src="i16")
    else:
        k.store("packed", at, word, src="i32")
    k.end()
    save(k)


quant(True)
quant(False)
quant(True, bits=8)
quant(False, bits=8)


def write():
    sources["sage_transpose"] = (ROOT / "kernels/sage_transpose.loom").read_text()
    out = ROOT / "kernels/native"
    out.mkdir(exist_ok=True)
    for name, src in sources.items():
        (out / (name + ".loom")).write_text(src)
    subprocess.run(
        [
            os.environ.get("LOOM_FORMAT", "loom-format"),
            "--in-place",
            *map(str, sorted(out.glob("*.loom"))),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    for name in sources:
        sources[name] = (out / (name + ".loom")).read_text()
    header = [
        "// Generated by tools/gen_native_ops.py.",
        "static const std::map<std::string, std::string> native_sources = {",
    ]
    for name, src in sources.items():
        header.append('{"' + name + '", R"LOOM(' + src + ')LOOM"},')
    header.append("};\n")
    path = ROOT / "host/native_sources.h"
    text = "\n".join(header)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


if __name__ == "__main__":
    write()
