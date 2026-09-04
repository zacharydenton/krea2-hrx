"""kernels/prepare_*_i4.loom: the GEMM-input preparation kernels, one workgroup of 256
lanes per token. Every variant ends the same way -- group-256 Hadamard (Kronecker
power of H4, normalised by 1/16), per-token absmax, symmetric int4 (q = round(x / s),
s = absmax/7), nibbles packed low first, f32 scale beside -- and differs in how the
row is formed first:
  norm   : (1 + mod_scale) * rmsnorm(h) * (1 + norm_scale) + mod_shift    (block inputs)
  gated  : sigmoid(gate) * attn                                            (the wo input)
  swiglu : silu(g) * u                                                     (the down input)
"""
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "kernels"

def stage(d: int, lds: str = "f32") -> str:
    return f"""  scf.for %t0 = [%lane to %quads step %c256] {{
    %t = index.assume %t0 [lt(%t0, %quads)] : index
    %g = index.div %t, %c64 : index
    %within = index.rem %t, %c64 : index
    %lo = index.rem %within, %c{d} : index
    %hi = index.div %within, %c{d} : index
    %hi4 = index.mul %hi, %c4 : index
    %hi4d = index.mul %hi4, %c{d} : index
    %g256 = index.mul %g, %c256 : index
    %base0 = index.add %g256, %hi4d : index
    %base = index.add %base0, %lo : index
    %i1 = index.add %base, %c{d} : index
    %i2 = index.add %i1, %c{d} : index
    %i3 = index.add %i2, %c{d} : index
    %e0 = view.load %x_view[%base] : view<[%width]xf32> -> f32
    %e1 = view.load %x_view[%i1] : view<[%width]xf32> -> f32
    %e2 = view.load %x_view[%i2] : view<[%width]xf32> -> f32
    %e3 = view.load %x_view[%i3] : view<[%width]xf32> -> f32
    %s01 = scalar.addf %e0, %e1 : f32
    %s23 = scalar.addf %e2, %e3 : f32
    %d01 = scalar.subf %e0, %e1 : f32
    %d23 = scalar.subf %e2, %e3 : f32
    %o0 = scalar.addf %s01, %d23 : f32
    %o1 = scalar.subf %s01, %d23 : f32
    %o2 = scalar.addf %d01, %s23 : f32
    %o3 = scalar.subf %s23, %d01 : f32
    view.store %o0, %x_view[%base] : f32, view<[%width]xf32>
    view.store %o1, %x_view[%i1] : f32, view<[%width]xf32>
    view.store %o2, %x_view[%i2] : f32, view<[%width]xf32>
    view.store %o3, %x_view[%i3] : f32, view<[%width]xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
"""

FORM = {
    "norm": dict(
        args="%h: buffer, %norm_scale: buffer, %mod_scale: buffer, %mod_shift: buffer",
        views="""  %h_global = buffer.assume.memory_space<global> %h : buffer
  %ns_global = buffer.assume.memory_space<global> %norm_scale : buffer
  %ms_global = buffer.assume.memory_space<global> %mod_scale : buffer
  %sh_global = buffer.assume.memory_space<global> %mod_shift : buffer
  %h_view = buffer.view %h_global[%c0_offset] : buffer -> view<[%tokens_b]x[%width]xf16>
  %ns_view = buffer.view %ns_global[%c0_offset] : buffer -> view<[%width]xf32>
  %ms_view = buffer.view %ms_global[%c0_offset] : buffer -> view<[%width]xf32>
  %sh_view = buffer.view %sh_global[%c0_offset] : buffer -> view<[%width]xf32>
""",
        form="""  // sum of squares, then normalise, scale and modulate into LDS
  %ss = scf.for %i0 = [%lane to %width step %c256](%acc = %zero : f32) -> (f32) {
    %i = index.assume %i0 [lt(%i0, %width)] : index
    %v16 = view.load %h_view[%row, %i] : view<[%tokens_b]x[%width]xf16> -> f16
    %v = scalar.extf %v16 : f16 to f32
    %sq = scalar.mulf %v, %v : f32
    %next = scalar.addf %acc, %sq : f32
    scf.yield %next : f32
  }
  %total = kernel.workgroup.reduce<addf> %ss : f32
  %width_i = index.cast %width : index to i32
  %width_f = scalar.sitofp %width_i : i32 to f32
  %mean = scalar.divf %total, %width_f : f32
  %mean_eps = scalar.addf %mean, %eps : f32
  %rms_inv = scalar.rsqrtf %mean_eps : f32
  scf.for %i0 = [%lane to %width step %c256] {
    %i = index.assume %i0 [lt(%i0, %width)] : index
    %v16 = view.load %h_view[%row, %i] : view<[%tokens_b]x[%width]xf16> -> f16
    %v = scalar.extf %v16 : f16 to f32
    %n = scalar.mulf %v, %rms_inv : f32
    %ns = view.load %ns_view[%i] : view<[%width]xf32> -> f32
    %ns1 = scalar.addf %ns, %one : f32
    %normed = scalar.mulf %n, %ns1 : f32
    %ms = view.load %ms_view[%i] : view<[%width]xf32> -> f32
    %ms1 = scalar.addf %ms, %one : f32
    %sh = view.load %sh_view[%i] : view<[%width]xf32> -> f32
    %scaled = scalar.mulf %normed, %ms1 : f32
    %modulated = scalar.addf %scaled, %sh : f32
    view.store %modulated, %x_view[%i] : f32, view<[%width]xf32>
  }
"""),
    "gated": dict(
        args="%attn: buffer, %gate: buffer",
        views="""  %a_global = buffer.assume.memory_space<global> %attn : buffer
  %g_global = buffer.assume.memory_space<global> %gate : buffer
  %a_view = buffer.view %a_global[%c0_offset] : buffer -> view<[%tokens_b]x[%width]xf16>
  %g_view = buffer.view %g_global[%c0_offset] : buffer -> view<[%tokens_b]x[%gate_stride]xf16>
""",
        form="""  scf.for %i0 = [%lane to %width step %c256] {
    %i = index.assume %i0 [lt(%i0, %width)] : index
    %a16 = view.load %a_view[%row, %i] : view<[%tokens_b]x[%width]xf16> -> f16
    %ig = index.assume %i [lt(%i, %gate_stride)] : index
    %g16 = view.load %g_view[%row, %ig] : view<[%tokens_b]x[%gate_stride]xf16> -> f16
    %a = scalar.extf %a16 : f16 to f32
    %g = scalar.extf %g16 : f16 to f32
    %neg_g = scalar.subf %zero, %g : f32
    %e = scalar.expf<afn> %neg_g : f32
    %den = scalar.addf %one, %e : f32
    %sig = scalar.divf %one, %den : f32
    %v = scalar.mulf %a, %sig : f32
    view.store %v, %x_view[%i] : f32, view<[%width]xf32>
  }
"""),
    "swiglu": dict(
        args="%gu: buffer",
        views="""  %gu_global = buffer.assume.memory_space<global> %gu : buffer
  %gu_view = buffer.view %gu_global[%c0_offset] : buffer -> view<[%tokens_b]x[%gate_stride]xf16>
""",
        form="""  // the fused gate|up GEMM output: gate at column i, up at column width + i
  scf.for %i0 = [%lane to %width step %c256] {
    %i = index.assume %i0 [lt(%i0, %width)] : index
    %iu0 = index.add %i, %width : index
    %iu = index.assume %iu0 [lt(%iu0, %gate_stride)] : index
    %ig = index.assume %i [lt(%i, %gate_stride)] : index
    %g16 = view.load %gu_view[%row, %ig] : view<[%tokens_b]x[%gate_stride]xf16> -> f16
    %u16 = view.load %gu_view[%row, %iu] : view<[%tokens_b]x[%gate_stride]xf16> -> f16
    %g = scalar.extf %g16 : f16 to f32
    %u = scalar.extf %u16 : f16 to f32
    %neg_g = scalar.subf %zero, %g : f32
    %e = scalar.expf<afn> %neg_g : f32
    %den = scalar.addf %one, %e : f32
    %sig = scalar.divf %one, %den : f32
    %silu = scalar.mulf %g, %sig : f32
    %v = scalar.mulf %silu, %u : f32
    view.store %v, %x_view[%i] : f32, view<[%width]xf32>
  }
"""),
}

def kernel(name: str) -> str:
    f = FORM[name]
    lds = "f16" if name == "swiglu" else "f32"
    lds_bytes = 2 if lds == "f16" else 4
    ns, sym = f"krea2.prepare_{name}_i4", f"krea2_prepare_{name}_i4"
    extra_cfg = "" if name == "norm" else f"\nconfig.decl @{ns}.gate_stride : %value: index where [range(%value, 256, 65536), mul(%value, 256)]\n"
    extra_get = "" if name == "norm" else f"  %gate_stride = config.get @{ns}.gate_stride : index\n"
    eps_cfg = f"\nconfig.decl @{ns}.eps : f32\n" if name == "norm" else ""
    eps_get = f"  %eps = config.get @{ns}.eps : f32\n" if name == "norm" else ""
    return f"""// GEMM input preparation ({name}), one workgroup of 256 lanes per token: form the
// row in f32 in LDS, rotate it by the group-256 Hadamard (H4 (x) H4 (x) H4 (x) H4 as four
// radix-4 stages of strides 1, 4, 16, 64; the 1/16 folds into the scale), take the
// token's absmax, and write symmetric int4 (q = round(x / s), s = absmax / 7, nibbles
// low first) with the f32 scale beside it -- the operand the int4 GEMM consumes.
//
// GENERATED by tools/gen_prepare.py; edit the generator.
amdgpu.target<gfx11-generic> @{sym}_gfx11 {{subgroup_size = 32}}

config.decl @{ns}.width : %value: index where [range(%value, 256, 32768), mul(%value, 256)]
{eps_cfg}{extra_cfg}
kernel.def target(@{sym}_gfx11) export("{sym}") @{sym}(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens: index, {f["args"]}, %q: buffer, %q_scale: buffer) {{
  %width = config.get @{ns}.width : index
{eps_get}{extra_get}  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c16 = index.constant 16 : index
  %c64 = index.constant 64 : index
  %c256 = index.constant 256 : index
  %c0_offset = index.constant 0 : offset
  %one = scalar.constant 1.0 : f32
  %zero = scalar.constant 0.0 : f32
  %seven = scalar.constant 7.0 : f32
  %neg_seven = scalar.constant -7.0 : f32
  %sixteenth = scalar.constant 0.0625 : f32
  %tiny = scalar.constant 1e-30 : f32
  %fifteen = scalar.constant 15 : i32
  %four = scalar.constant 4 : i32
  %tokens_b = index.assume %tokens [range(%tokens, 1, 1048576)] : index
  %token = kernel.workgroup.id<x> : index
  %row = index.assume %token [lt(%token, %tokens_b)] : index
  %lane = kernel.workitem.id<x> : index
  %half_width = index.div %width, %c2 : index
  %quads = index.div %width, %c4 : index
{f["views"]}  %q_global = buffer.assume.memory_space<global> %q : buffer
  %qs_global = buffer.assume.memory_space<global> %q_scale : buffer
  %q_view = buffer.view %q_global[%c0_offset] : buffer -> view<[%tokens_b]x[%half_width]xi8>
  %qs_view = buffer.view %qs_global[%c0_offset] : buffer -> view<[%tokens_b]xf32>
  %row_bytes0 = index.mul %width, %c{lds_bytes} : index
  %row_bytes = index.cast %row_bytes0 : index to offset
  %lds = buffer.alloca<workgroup> align(16) %row_bytes : buffer
  %x_view = buffer.view %lds[%c0_offset] : buffer -> view<[%width]x{lds}>

{f["form"]}  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)

  // Hadamard: four radix-4 stages; each lane owns whole quads, so no barrier inside a stage.
{stage(1, lds)}{stage(4, lds)}{stage(16, lds)}{stage(64, lds)}
  // absmax (the 1/16 normalisation folded into the scale), quantise, pack
  %amax = scf.for %i0 = [%lane to %width step %c256](%acc = %zero : f32) -> (f32) {{
    %i = index.assume %i0 [lt(%i0, %width)] : index
    %v = view.load %x_view[%i] : view<[%width]xf32> -> f32
    %a = scalar.absf %v : f32
    %next = scalar.maxnumf %acc, %a : f32
    scf.yield %next : f32
  }}
  %row_max0 = kernel.workgroup.reduce<maxnumf> %amax : f32
  %row_max = scalar.mulf %row_max0, %sixteenth : f32
  %row_max_safe = scalar.maxnumf %row_max, %tiny : f32
  %s = scalar.divf %row_max_safe, %seven : f32
  %inv_s = scalar.divf %sixteenth, %s : f32
  scf.for %b0 = [%lane to %half_width step %c256] {{
    %b = index.assume %b0 [lt(%b0, %half_width)] : index
    %i0 = index.mul %b, %c2 : index
    %i1 = index.add %i0, %c1 : index
    %v0 = view.load %x_view[%i0] : view<[%width]xf32> -> f32
    %v1 = view.load %x_view[%i1] : view<[%width]xf32> -> f32
    %r0 = scalar.mulf %v0, %inv_s : f32
    %r1 = scalar.mulf %v1, %inv_s : f32
    %q0f = scalar.roundevenf %r0 : f32
    %q1f = scalar.roundevenf %r1 : f32
    %q0c = scalar.maxnumf %q0f, %neg_seven : f32
    %q1c = scalar.maxnumf %q1f, %neg_seven : f32
    %q0d = scalar.minnumf %q0c, %seven : f32
    %q1d = scalar.minnumf %q1c, %seven : f32
    %q0 = scalar.fptosi %q0d : f32 to i32
    %q1 = scalar.fptosi %q1d : f32 to i32
    %lo_n = scalar.andi %q0, %fifteen : i32
    %hi_n0 = scalar.andi %q1, %fifteen : i32
    %hi_n = scalar.shli %hi_n0, %four : i32
    %byte32 = scalar.ori %lo_n, %hi_n : i32
    %byte = scalar.trunci %byte32 : i32 to i8
    view.store %byte, %q_view[%row, %b] : i8, view<[%tokens_b]x[%half_width]xi8>
  }}
  // the token scale, by one lane, after the last loop (a divergent region before a
  // loop is rejected by the branch lowering)
  %is_first = index.cmp eq, %lane, %c0 : index
  scf.if %is_first {{
    view.store %s, %qs_view[%row] : f32, view<[%tokens_b]xf32>
  }}
  kernel.return
}}
"""


def lds_type(text: str, lds: str) -> str:
    """Route every LDS row access through the chosen element type: f16 storage with
    f32 arithmetic (extf on load, fptrunc on store)."""
    if lds == "f32":
        return text
    text = text.replace("view<[%width]xf32>", "view<[%width]xf16>")
    out = []
    for line in text.split("\n"):
        m = re.match(r"^(\s*)(%\w+) = view\.load %x_view\[(%\w+)\] : view<\[%width\]xf16> -> f32$", line)
        if m:
            ind, name, idx = m.groups()
            out.append(f"{ind}{name}_h = view.load %x_view[{idx}] : view<[%width]xf16> -> f16")
            out.append(f"{ind}{name} = scalar.extf {name}_h : f16 to f32")
            continue
        m = re.match(r"^(\s*)view\.store (%\w+), %x_view\[(%\w+)\] : f32, view<\[%width\]xf16>$", line)
        if m:
            ind, name, idx = m.groups()
            out.append(f"{ind}{name}_h = scalar.fptrunc {name} : f32 to f16")
            out.append(f"{ind}view.store {name}_h, %x_view[{idx}] : f16, view<[%width]xf16>")
            continue
        out.append(line)
    return "\n".join(out)


def uniform_loops(text: str) -> str:
    """`scf.for %x0 = [%lane to %bound step %c256]` has a lane-dependent entry, which the
    branch lowering rejects; every lane runs bound/256 iterations, so loop over that
    count and add the lane inside."""
    counts = {"%width": "%width_per_lane", "%quads": "%quads_per_lane", "%half_width": "%half_per_lane"}
    def repl(m):
        var, bound, carried = m.group(1), m.group(2), m.group(3) or ""
        return (f"scf.for %{var}_j = [%c0 to {counts[bound]} step %c1]{carried} {{\n"
                f"    %{var}_lane_step = index.mul %{var}_j, %c256 : index\n"
                f"    %{var}0 = index.add %{var}_lane_step, %lane : index\n")
    text = re.sub(r"scf\.for %(\w+)0 = \[%lane to (%\w+) step %c256\]((?:\(.*?\) -> \(.*?\))?) \{\n", repl, text)
    return text.replace("  %quads = index.div %width, %c4 : index\n",
                        "  %quads = index.div %width, %c4 : index\n  %width_per_lane = index.div %width, %c256 : index\n"
                        "  %quads_per_lane = index.div %quads, %c256 : index\n  %half_per_lane = index.div %half_width, %c256 : index\n")

for name in FORM:
    (OUT / f"prepare_{name}_i4.loom").write_text(uniform_loops(lds_type(kernel(name), "f16" if name == "swiglu" else "f32")))
    print("wrote", f"prepare_{name}_i4.loom")
