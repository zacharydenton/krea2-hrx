"""kernels/gemm_i{4,8}{,_resid,_swiglu}_256.loom: Krea 2's W4A4 and W8A8 GEMM families on the
256x128 workgroup tile (eight 64x64 wave tiles, 0.5 LDS operand reads per multiply), with the
three epilogues the blocks need, the int4 arithmetic identical to the 128x128 kernels:
  plain  : C[m, n] = f16( acc * w_scale[n] * a_scale[m] )
  resid  : x[m, n] = f16( x[m, n] + gate[n] * acc * w_scale[n] * a_scale[m] )   (in place on the f16 stream)
  swiglu : C[m, o] = f16( silu(g) * u ), weight rows interleaved in 16-row gate/up groups
A [M][K] and W [N][K] are int4 nibbles (low first) or int8 bytes, i32 accumulation, f32 scales.

    python3 tools/gen_gemm.py            writes the six kernels (then loom-format them)
Ported from MiniMax H3's tools/gen_gemm.py (loom-gemm's 256x128 tile); GENERATED kernels,
edit the generator.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TM, TN = 256, 128
WM, WN = 64, 64
FM, FN = WM // 16, WN // 16
V2, V4, V8 = "vector<2xi32>", "vector<4xi32>", "vector<8xi32>"
ACC = [f"{i}{j}" for i in range(FM) for j in range(FN)]
PACKETS = (TM + TN) * 4 // 256
A_PACKETS = TM * 4 // 256


def generate(mode: str, loads: str = "plain", raster: str = "shorten", bits: int = 4) -> str:
    """bits: 4 (int4 nibbles, 128-wide k steps) or 8 (int8 bytes, 64-wide k steps); a k step is 64 bytes
    per row either way, so the LDS stage and the staging map are the same.
    raster: "shorten" launches exactly ceil(M/256) tile rows and the last raster group holds the
    remaining tiles (no ghost workgroups; m_group is the full group size); "pad" launches the
    tile rows rounded up to a multiple of m_group, the H3 form (ghost tile rows run the whole K loop).
    loads: where the next step's global packets are issued.
    "plain"      unconditionally, after the barrier (rows past M are clamped to row 0, k past K to
                 the first quad; neither is ever published or staged). Loom sinks the loads into
                 the WMMA block; on the 128x128 tile that cost the K=16384 down projection 3-4%
                 and gained the K=6144 wo projection 4%, so each epilogue is timed separately.
    "first"      unconditionally, before the current packets are stored to LDS, ahead of the
                 barrier (0.88-0.92x on the 128x128 tile: falsified there).
    "predicated" the 128x128 kernels' original scf.if form (zero past M and K)."""
    assert mode in ("plain", "resid", "swiglu") and loads in ("first", "plain", "predicated") and raster in ("shorten", "pad") and bits in (4, 8)
    # k per step, k per i32 quad, 16-wide sub-steps per step, fragment payload type and registers
    KSTEP, KQ, SUBS, FRAG, PREGS = (128, 8, 8, V2, 2) if bits == 4 else (64, 4, 4, V4, 4)
    STEM = {"plain": f"gemm_i{bits}_256", "resid": f"gemm_i{bits}_resid_256", "swiglu": f"gemm_i{bits}_swiglu_256"}[mode]
    SYM, NS = "krea2_" + STEM, "krea2." + STEM
    extra_args = ", %gate: buffer" if mode == "resid" else ""
    if raster == "pad":
        GRID = f"""  %m_group = config.get @{NS}.m_group : index
  %m_group_less = index.sub %m_group, %c1 : index
  %m_tiles3 = index.add %m_tiles, %m_group_less : index
  %m_groups = index.div %m_tiles3, %m_group : index
  %m_tiles_launched = index.mul %m_groups, %m_group : index
"""
        RASTER = f"""  // Rasterization: the launch is n_tiles x (m_tiles rounded up to a multiple of m_group),
  // and tiles are visited in groups of m_group m-tiles per sweep of n, so each W tile is
  // read m_group times in quick succession from cache instead of once per row of tiles.
  // A tile row past M is a ghost: its A rows are clamped to row 0 (loaded, never published).
  %wg_x = kernel.workgroup.id<x> : index
  %wg_y = kernel.workgroup.id<y> : index
  %c_n_tiles = index.div %n_size, %c128 : index
  %linear0 = index.mul %wg_y, %c_n_tiles : index
  %linear = index.add %linear0, %wg_x : index
  %m_group = config.get @{NS}.m_group : index
  %group_span = index.mul %c_n_tiles, %m_group : index
  %group = index.div %linear, %group_span : index
  %in_group = index.rem %linear, %group_span : index
  %group_m = index.mul %group, %m_group : index
  %m_in_group = index.rem %in_group, %m_group : index
  %tile_m_id = index.add %group_m, %m_in_group : index
  %tile_n_id0 = index.div %in_group, %m_group : index
  %tile_n_id = index.assume %tile_n_id0 [lt(%tile_n_id0, %c_n_tiles)] : index
"""
    else:
        GRID = "  %m_tiles_launched = index.add %m_tiles, %c0 : index\n"
        RASTER = f"""  // Rasterization: the launch is n_tiles x ceil(M/256), and tiles are visited in groups
  // of m_group m-tiles per sweep of n, so each W tile is read m_group times in quick
  // succession from cache instead of once per row of tiles. The last group holds the
  // remaining tiles, so no ghost workgroups run.
  %wg_x = kernel.workgroup.id<x> : index
  %wg_y = kernel.workgroup.id<y> : index
  %c_n_tiles = index.div %n_size, %c128 : index
  %linear0 = index.mul %wg_y, %c_n_tiles : index
  %linear = index.add %linear0, %wg_x : index
  %m_group = config.get @{NS}.m_group : index
  %group_span = index.mul %c_n_tiles, %m_group : index
  %group = index.div %linear, %group_span : index
  %in_group = index.rem %linear, %group_span : index
  %group_m = index.mul %group, %m_group : index
  %last_add = index.add %m_bounded, %c255 : index
  %real_tiles = index.div %last_add, %c256 : index
  %remaining = index.sub %real_tiles, %group_m : index
  %short_group = index.cmp ult, %remaining, %m_group : index
  %group_rows0 = scf.select %short_group, %remaining, %m_group : index
  %group_rows = index.assume %group_rows0 [range(%group_rows0, 1, 4)] : index
  %g_is1 = index.cmp eq, %group_rows, %c1 : index
  %g_is2 = index.cmp eq, %group_rows, %c2 : index
  %g_is3 = index.cmp eq, %group_rows, %c3 : index
  %q2 = index.div %in_group, %c2 : index
  %q3 = index.div %in_group, %c3 : index
  %q4 = index.div %in_group, %c4 : index
  %q34 = scf.select %g_is3, %q3, %q4 : index
  %q234 = scf.select %g_is2, %q2, %q34 : index
  %real_n = scf.select %g_is1, %in_group, %q234 : index
  %col_group_offset = index.mul %real_n, %group_rows : index
  %real_m0 = index.sub %in_group, %col_group_offset : index
  %m_in_group = index.assume %real_m0 [range(%real_m0, 0, 3)] : index
  %tile_m_id = index.add %group_m, %m_in_group : index
  %tile_n_id0 = index.add %real_n, %c0 : index
  %tile_n_id = index.assume %tile_n_id0 [lt(%tile_n_id0, %c_n_tiles)] : index
"""
    head = {
        "plain": "// Krea 2's int4 GEMM on the 256x128 tile: the W4A4 ConvRot epilogue scales the i32\n"
                 "// accumulator by the weight row's scale and the activation token's scale:\n"
                 "//   C[m, n] = f16( acc[m, n] * w_scale[n] * a_scale[m] )\n",
        "resid": "// Krea 2's residual int4 GEMM on the 256x128 tile: the scaled accumulator is gated per\n"
                 "// column and added in place to the f16 residual stream:\n"
                 "//   x[m, n] = f16( x[m, n] + gate[n] * acc[m, n] * w_scale[n] * a_scale[m] )\n",
        "swiglu": "// Krea 2's gate|up int4 GEMM on the 256x128 tile with the SwiGLU product in the\n"
                  "// epilogue: weight rows are interleaved in 16-row groups [gate o..o+15 | up o..o+15], so\n"
                  "// every wave holds the gate and up fragments of the same 16 outputs side by side and writes\n"
                  "//   C[m, o] = f16( silu(g) * u ),  g = acc_gate * w_scale * a_scale,  u = acc_up * w_scale * a_scale\n"
                  "// to a [M][N/2] output.\n",
    }[mode]
    K = head + f"""// A [M][K] and W [N][K] are {"int4 nibbles (low first)" if bits == 4 else "int8 bytes"}, both Hadamard-rotated along K.
// Workgroup tile {TM}x{TN}, eight waves as 4 (m) x 2 (n) of {WM}x{WN} (4 x 4 accumulator fragments);
// per 16-wide k sub-step a wave loads 4 A and 4 W runs from the stage and issues 16 WMMAs.
// One {(TM + TN) * 80}-byte stage ({TM} A rows + {TN} W rows of 80 bytes), packets carried one step
// ahead in registers, grouped rasterization (m_group m-tiles per n sweep).{" The arithmetic and" if bits == 4 else ""}
{"// FP16 rounding match the 128x128 kernel of the same epilogue exactly." if bits == 4 else "// The epilogue arithmetic is the int4 family's."}
//
// GENERATED by tools/gen_gemm.py; edit the generator.
amdgpu.target<gfx11-generic> @{SYM}_gfx11 {{subgroup_size = 32}}

config.decl @{NS}.k_size : %value: index where [range(%value, {KSTEP}, 65536), mul(%value, {KSTEP})]

config.decl @{NS}.n_size : %value: index where [range(%value, 128, 32768), mul(%value, 128)]

// operand row pitch in k elements: K, or K plus a step when the row's bytes are a multiple of 1024
config.decl @{NS}.k_stride : %value: index where [range(%value, {KSTEP}, 65536), mul(%value, {KSTEP})]

// m-tiles per raster group: chosen per token count so the tile rows divide with as few ghost rows as possible
config.decl @{NS}.m_group : %value: index where [range(%value, 1, 4)]

kernel.def target(@{SYM}_gfx11) export("{SYM}") @{SYM}(%m_size: index) {{
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c128 = index.constant 128 : index
  %c255 = index.constant 255 : index
  %c256 = index.constant 256 : index
  %n_size0 = config.get @{NS}.n_size : index
  %m_rounded = index.add %m_size, %c255 : index
  %m_tiles = index.div %m_rounded, %c256 : index
{GRID}  %n_tiles = index.div %n_size0, %c128 : index
  kernel.launch.config workgroups(%n_tiles, %m_tiles_launched, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%m_size: index, %a: buffer, %w: buffer, %scale: buffer, %a_scale: buffer, %c: buffer{extra_args}) {{
  %k_size0 = config.get @{NS}.k_size : index
  %n_size0 = config.get @{NS}.n_size : index
  %k_stride0 = config.get @{NS}.k_stride : index
  %k_size = index.assume %k_size0 [range(%k_size0, {KSTEP}, 65536), mul(%k_size0, {KSTEP})] : index
  %n_size = index.assume %n_size0 [range(%n_size0, 128, 32768), mul(%n_size0, 128)] : index
  %k_stride1 = index.assume %k_stride0 [range(%k_stride0, {KSTEP}, 65536), mul(%k_stride0, {KSTEP})] : index
  %k_stride = index.assume %k_stride1 [ge(%k_stride1, %k_size)] : index

"""
    for c in (0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 32, 48, 64, 128, 192, 255, 256):
        K += f"  %c{c} = index.constant {c} : index\n"
    for f in range(16):
        K += f"  %cf{f} = index.constant {f} : index\n"
    K += f"""  %c0_offset = index.constant 0 : offset
  %lds_bytes = index.constant {(TM + TN) * 80} : offset
  %w_stage_offset = index.constant {TM * 80} : offset
  %wave_result_bytes = index.constant 2048 : offset
  %m = index.constant 16 : index
  %n = index.constant 16 : index
  %k = index.constant 16 : index
  %zero_i32x8 = vector.constant 0 : vector<8xi32>
"""
    if loads == "predicated":
        K += "  %zero_i32x4 = vector.constant 0 : vector<4xi32>\n"
    K += f"""  %i4_schema = encoding.define #encoding.operand<element_format=i{bits}, payload_elements=16, payload_registers={PREGS}> : encoding<schema>

  %workitem = kernel.workitem.id<x> : index
  %subgroup0 = kernel.subgroup.id : index
  %subgroup = index.assume %subgroup0 [range(%subgroup0, 0, 7)] : index
  %lane = kernel.subgroup.lane.id : index
  %lane16 = index.rem %lane, %c16 : index
  %m_bounded = index.assume %m_size [range(%m_size, 1, 16777216)] : index
{RASTER}
  // Wave grid 4 (m) x 2 (n): rows wave_row..+{WM}, columns wave_col..+{WN} of the tile.
  %wave_m = index.div %subgroup, %c2 : index
  %wave_n = index.rem %subgroup, %c2 : index
  %wave_row = index.mul %wave_m, %c{WM} : index
  %wave_col = index.mul %wave_n, %c{WN} : index
  %base_m = index.mul %tile_m_id, %c{TM} : index
  %base_n = index.mul %tile_n_id, %c{TN} : index
  // A and W rows are k_stride/{KQ} i32 quads; only the first k_size/{KQ} are read.
  %k_quads = index.div %k_stride, %c{KQ} : index

  %a_global = buffer.assume.memory_space<global> %a : buffer
  %w_global = buffer.assume.memory_space<global> %w : buffer
  %scale_global = buffer.assume.memory_space<global> %scale : buffer
  %a_scale_global = buffer.assume.memory_space<global> %a_scale : buffer
  %c_global = buffer.assume.memory_space<global> %c : buffer
"""
    if mode == "resid":
        K += "  %gate_global = buffer.assume.memory_space<global> %gate : buffer\n"
    K += """  %a_view = buffer.view %a_global[%c0_offset] : buffer -> view<[%m_bounded]x[%k_quads]xi32>
  %w_view = buffer.view %w_global[%c0_offset] : buffer -> view<[%n_size]x[%k_quads]xi32>
  %scale_view = buffer.view %scale_global[%c0_offset] : buffer -> view<[%n_size]xf32>
  %a_scale_view = buffer.view %a_scale_global[%c0_offset] : buffer -> view<[%m_bounded]xf32>
"""
    if mode == "plain":
        K += "  %c_view = buffer.view %c_global[%c0_offset] : buffer -> view<[%m_bounded]x[%n_size]xf16>\n"
    elif mode == "resid":
        K += "  %c_view = buffer.view %c_global[%c0_offset] : buffer -> view<[%m_bounded]x[%n_size]xf16>\n"
        K += "  %gate_view = buffer.view %gate_global[%c0_offset] : buffer -> view<[%n_size]xf32>\n"
    else:
        K += "  %n_half = index.div %n_size, %c2 : index\n  %c_view = buffer.view %c_global[%c0_offset] : buffer -> view<[%m_bounded]x[%n_half]xf16>\n"
    K += f"""
  %lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer
  // Stages: rows of 20 quads (64 bytes of k, 16 of padding against bank conflicts).
  %a_stage = buffer.view %lds[%c0_offset] : buffer -> view<{TM}x20xi32>
  %w_stage = buffer.view %lds[%w_stage_offset] : buffer -> view<{TN}x20xi32>
  %wave_result_offset = index.scale %subgroup, %wave_result_bytes : index, offset -> offset
  %result_view = buffer.view %lds[%wave_result_offset] : buffer -> view<16x16xi32>
"""
    if mode == "swiglu":
        K += """  %up_stage_offset = index.constant 1024 : offset
  %wave_up_offset = index.add %wave_result_offset, %up_stage_offset : offset
  %up_view = buffer.view %lds[%wave_up_offset] : buffer -> view<16x16xi32>
"""
    K += f"""
  // Staging map: packet p = workitem + 256*i covers row p/4, quad column (p%4)*4;
  // i = 0..{A_PACKETS - 1} are A rows 0..{TM - 1}, i = {A_PACKETS}..{PACKETS - 1} are W rows 0..{TN - 1}.
  %st_sub = index.rem %workitem, %c4 : index
  %st_quad = index.mul %st_sub, %c4 : index
  %st_row_base = index.div %workitem, %c4 : index
  // A step is 16 quads; this lane's slot within it is st_quad. Global columns are
  // step + slot, bounded by K/4 - 4 and 4-aligned.
  %k_quad_limit = index.sub %k_quads, %c4 : index
"""
    for i in range(A_PACKETS):
        K += f"  %st_arow{i} = index.add %st_row_base, %c{64 * i} : index\n"
        K += f"  %a_m{i} = index.add %base_m, %st_arow{i} : index\n  %a_ok{i} = index.cmp ult, %a_m{i}, %m_bounded : index\n"
        K += f"  %a_safe{i} = scf.select %a_ok{i}, %a_m{i}, %c0 : index\n  %a_row{i} = index.assume %a_safe{i} [lt(%a_safe{i}, %m_bounded)] : index\n"
    for i in range(PACKETS - A_PACKETS):
        K += f"  %st_wrow{i} = index.add %st_row_base, %c{64 * i} : index\n"
        K += f"  %w_n{i} = index.add %base_n, %st_wrow{i} : index\n  %w_row{i} = index.assume %w_n{i} [lt(%w_n{i}, %n_size)] : index\n"
    K += "\n  // Operand rows this lane reads: A rows wave_row + 16i + lane16, W rows wave_col + 16j + lane16.\n"
    for i in range(FM):
        K += f"  %fa{i} = index.add %wave_row, %lane16 : index\n" if i == 0 else f"  %fa{i} = index.add %fa0, %c{16 * i} : index\n"
    for j in range(FN):
        K += f"  %fb{j} = index.add %wave_col, %lane16 : index\n" if j == 0 else f"  %fb{j} = index.add %fb0, %c{16 * j} : index\n"
    K += """
  %acc_init = vector.fragment<init> %zero_i32x8 shape [%m, %n] : vector<8xi32>

  // Gather of the first k step (64 bytes = 16 quads) into registers.
  %kq_first = index.assume %st_quad [le(%st_quad, %k_quad_limit), mul(%st_quad, 4)] : index
"""
    def a_load(i, name, col, cond, indent):
        if loads != "predicated":
            return f"{indent}%{name} = vector.load %a_view[%a_row{i}, {col}] : view<[%m_bounded]x[%k_quads]xi32> -> {V4}\n"
        return (f"{indent}%{name} = scf.if {cond} -> ({V4}) {{\n{indent}  %v = vector.load %a_view[%a_row{i}, {col}] : view<[%m_bounded]x[%k_quads]xi32> -> {V4}\n"
                f"{indent}  scf.yield %v : {V4}\n{indent}}} else {{\n{indent}  scf.yield %zero_i32x4 : {V4}\n{indent}}}\n")
    def w_load(i, name, col, cond, indent):
        if loads != "predicated":
            return f"{indent}%{name} = vector.load %w_view[%w_row{i}, {col}] : view<[%n_size]x[%k_quads]xi32> -> {V4}\n"
        return (f"{indent}%{name} = scf.if {cond} -> ({V4}) {{\n{indent}  %v = vector.load %w_view[%w_row{i}, {col}] : view<[%n_size]x[%k_quads]xi32> -> {V4}\n"
                f"{indent}  scf.yield %v : {V4}\n{indent}}} else {{\n{indent}  scf.yield %zero_i32x4 : {V4}\n{indent}}}\n")
    for i in range(A_PACKETS):
        K += a_load(i, f"ga{i}", "%kq_first", f"%a_ok{i}", "  ")
    for i in range(PACKETS - A_PACKETS):
        K += w_load(i, f"gw{i}", "%kq_first", "%c1_true", "  ") if loads != "predicated" else f"  %gw{i} = vector.load %w_view[%w_row{i}, %kq_first] : view<[%n_size]x[%k_quads]xi32> -> {V4}\n"
    carried = ", ".join([f"%acc{x}" for x in ACC] + [f"%a_last{i}" for i in range(A_PACKETS)] + [f"%w_last{i}" for i in range(PACKETS - A_PACKETS)])
    inits = ", ".join([f"%r{x} = %acc_init : {V8}" for x in ACC] + [f"%ca{i} = %ga{i} : {V4}" for i in range(A_PACKETS)] + [f"%cw{i} = %gw{i} : {V4}" for i in range(PACKETS - A_PACKETS)])
    types = ", ".join([V8] * len(ACC) + [V4] * PACKETS)
    K += f"\n  {carried} = scf.for %k_base = [%c0 to %k_size step %c{KSTEP}]({inits}) -> ({types}) {{\n"
    prefetch = f"""    // Prefetch the next step while the WMMAs run on this one.
    %k_next = index.add %k_base, %c{KSTEP} : index
    %in_k = index.cmp ult, %k_next, %k_size : index
    %kq_next0 = index.div %k_next, %c{KQ} : index
    %kq_slot = index.add %kq_next0, %st_quad : index
    %kq_safe = scf.select %in_k, %kq_slot, %st_quad : index
    %kq_next = index.assume %kq_safe [le(%kq_safe, %k_quad_limit), mul(%kq_safe, 4)] : index
"""
    if loads == "predicated":
        for i in range(A_PACKETS):
            prefetch += f"    %load_a{i} = scalar.andi %in_k, %a_ok{i} : i1\n"
    for i in range(A_PACKETS):
        prefetch += a_load(i, f"na{i}", "%kq_next", f"%load_a{i}", "    ")
    for i in range(PACKETS - A_PACKETS):
        prefetch += w_load(i, f"nw{i}", "%kq_next", "%in_k", "    ")
    stage = "    // Stage what we were handed.\n"
    for i in range(A_PACKETS):
        stage += f"    vector.store %ca{i}, %a_stage[%st_arow{i}, %st_quad] : {V4}, view<{TM}x20xi32>\n"
    for i in range(PACKETS - A_PACKETS):
        stage += f"    vector.store %cw{i}, %w_stage[%st_wrow{i}, %st_quad] : {V4}, view<{TN}x20xi32>\n"
    stage += "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
    if loads == "first":
        K += prefetch.replace("while the WMMAs run on this one", "ahead of the barrier, so the whole step covers it") + "\n" + stage
    else:
        K += stage + "\n" + prefetch
    K += f"\n    // {SUBS} 16-wide sub-steps over the staged 64 bytes: 4 A + 4 W runs, 16 WMMAs each.\n"
    prev = {x: f"%r{x}" for x in ACC}
    for s in range(SUBS):
        for i in range(FM):
            K += f"    %da{i}_{s} = vector.load %a_stage[%fa{i}, %c{PREGS * s}] : view<{TM}x20xi32> -> {FRAG}\n"
        for j in range(FN):
            K += f"    %db{j}_{s} = vector.load %w_stage[%fb{j}, %c{PREGS * s}] : view<{TN}x20xi32> -> {FRAG}\n"
        for i in range(FM):
            K += f"    %la{i}_{s} = vector.fragment<lhs> %da{i}_{s} shape [%m, %k] using {{schema = %i4_schema : encoding<schema>}} : {FRAG}\n"
        for j in range(FN):
            K += f"    %rb{j}_{s} = vector.fragment<rhs> %db{j}_{s} shape [%k, %n] using {{schema = %i4_schema : encoding<schema>}} : {FRAG}\n"
        for i in range(FM):
            for j in range(FN):
                x = f"{i}{j}"
                K += f"    %o{x}_{s} = vector.mma %la{i}_{s}, %rb{j}_{s}, {prev[x]} : {FRAG}, {FRAG}, {V8}\n"
                prev[x] = f"%o{x}_{s}"
    yields = ", ".join([prev[x] for x in ACC] + [f"%na{i}" for i in range(A_PACKETS)] + [f"%nw{i}" for i in range(PACKETS - A_PACKETS)])
    K += f"""    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    scf.yield {yields} : {types}
  }}

  // Publish through LDS: each lane takes two vector4 slices of each staged 16x16 tile.
  %publish_row0 = index.div %lane, %c4 : index
  %publish_col_group = index.rem %lane, %c4 : index
  %publish_col = index.mul %publish_col_group, %c4 : index
"""
    def chain(name, pick, count):
        text, sel = "", None
        for f in range(count):
            if sel is None:
                sel = f"%acc{pick(f)}"
                continue
            text += f"    %{name}_is{f} = index.cmp eq, %step, %cf{f} : index\n    %{name}_p{f} = scf.select %{name}_is{f}, %acc{pick(f)}, {sel} : {V8}\n"
            sel = f"%{name}_p{f}"
        return text, sel
    if mode in ("plain", "resid"):
        K += "\n  scf.for %step = [%c0 to %c16 step %c1] {\n    %fi = index.div %step, %c4 : index\n    %fj = index.rem %step, %c4 : index\n"
        text, sel = chain("g", lambda f: ACC[f], 16)
        K += text
        K += f"""    vector.fragment.store<result> {sel}, %result_view[%c0, %c0] shape [%m, %n] : {V8}, view<16x16xi32>
    kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)

    %frag_row_offset = index.mul %fi, %c16 : index
    %frag_col_offset = index.mul %fj, %c16 : index
    %tile_col = index.add %wave_col, %frag_col_offset : index
    %out_col_local = index.add %tile_col, %publish_col : index
    %out_col = index.add %base_n, %out_col_local : index
    %scale_values = vector.load %scale_view[%out_col] : view<[%n_size]xf32> -> vector<4xf32>
"""
        if mode == "resid":
            K += "    %gate_values = vector.load %gate_view[%out_col] : view<[%n_size]xf32> -> vector<4xf32>\n"
        K += """
    scf.for %half = [%c0 to %c2 step %c1] {
      %row_offset = index.mul %half, %c8 : index
      %publish_row = index.add %publish_row0, %row_offset : index
      %values = vector.load %result_view[%publish_row, %publish_col] : view<16x16xi32> -> vector<4xi32>
      %values_f32 = vector.sitofp %values : vector<4xi32> to vector<4xf32>
      %scaled0 = vector.mulf %values_f32, %scale_values : vector<4xf32>
      %row_in_wave = index.add %frag_row_offset, %publish_row : index
      %row_local = index.add %wave_row, %row_in_wave : index
      %out_row = index.add %base_m, %row_local : index
      %writes = index.cmp ult, %out_row, %m_bounded : index
      scf.if %writes {
        %bounded = index.assume %out_row [lt(%out_row, %m_bounded)] : index
        %a_scale_value = view.load %a_scale_view[%bounded] : view<[%m_bounded]xf32> -> f32
        %a_scale_vector = vector.splat %a_scale_value : vector<4xf32>
        %scaled = vector.mulf %scaled0, %a_scale_vector : vector<4xf32>
"""
        if mode == "plain":
            K += """        %narrow = vector.fptrunc %scaled : vector<4xf32> to vector<4xf16>
"""
        else:
            K += """        %gated = vector.mulf %scaled, %gate_values : vector<4xf32>
        %prior_half = vector.load %c_view[%bounded, %out_col] : view<[%m_bounded]x[%n_size]xf16> -> vector<4xf16>
        %prior = vector.extf %prior_half : vector<4xf16> to vector<4xf32>
        %summed = vector.addf %prior, %gated : vector<4xf32>
        %narrow = vector.fptrunc %summed : vector<4xf32> to vector<4xf16>
"""
        K += """        vector.store %narrow, %c_view[%bounded, %out_col] : vector<4xf16>, view<[%m_bounded]x[%n_size]xf16>
      }
    }
    kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)
  }
"""
    else:
        K += """  %publish_col_half = index.mul %publish_col_group, %c2 : index
  %one_f32 = scalar.constant 1.0 : f32
  %zero_f32_e = scalar.constant 0.0 : f32
  %one4 = vector.splat %one_f32 : vector<4xf32>
  %zero4 = vector.splat %zero_f32_e : vector<4xf32>

  // Per pair of fragments (gate fj, up fj+1) of the same 16 outputs, both staged,
  // then each lane takes two vector4 slices of each.
  scf.for %step = [%c0 to %c8 step %c1] {
    %fi = index.div %step, %c2 : index
    %pj = index.rem %step, %c2 : index
"""
        tg, sg = chain("g", lambda f: ACC[(f // 2) * 4 + (f % 2) * 2], 8)
        tu, su = chain("u", lambda f: ACC[(f // 2) * 4 + (f % 2) * 2 + 1], 8)
        K += tg + tu
        K += f"""    vector.fragment.store<result> {sg}, %result_view[%c0, %c0] shape [%m, %n] : {V8}, view<16x16xi32>
    vector.fragment.store<result> {su}, %up_view[%c0, %c0] shape [%m, %n] : {V8}, view<16x16xi32>
    kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)

    %frag_row_offset = index.mul %fi, %c16 : index
    %gate_col_offset = index.mul %pj, %c32 : index
    %gate_tile_col = index.add %wave_col, %gate_col_offset : index
    %gate_col_local = index.add %gate_tile_col, %publish_col : index
    %gate_col = index.add %base_n, %gate_col_local : index
    %up_col = index.add %gate_col, %c16 : index
    %gate_scale = vector.load %scale_view[%gate_col] : view<[%n_size]xf32> -> vector<4xf32>
    %up_scale = vector.load %scale_view[%up_col] : view<[%n_size]xf32> -> vector<4xf32>
    // output column: 16 outputs per 32 interleaved input columns
    %out_col0 = index.div %gate_col, %c2 : index
    %out_col1 = index.add %out_col0, %publish_col : index
    %out_col2 = index.sub %out_col1, %publish_col_half : index
    %out_col = index.assume %out_col2 [lt(%out_col2, %n_half)] : index

    scf.for %half = [%c0 to %c2 step %c1] {{
      %row_offset = index.mul %half, %c8 : index
      %publish_row = index.add %publish_row0, %row_offset : index
      %gate_values = vector.load %result_view[%publish_row, %publish_col] : view<16x16xi32> -> vector<4xi32>
      %up_values = vector.load %up_view[%publish_row, %publish_col] : view<16x16xi32> -> vector<4xi32>
      %gate_f32 = vector.sitofp %gate_values : vector<4xi32> to vector<4xf32>
      %up_f32 = vector.sitofp %up_values : vector<4xi32> to vector<4xf32>
      %gate_scaled0 = vector.mulf %gate_f32, %gate_scale : vector<4xf32>
      %up_scaled0 = vector.mulf %up_f32, %up_scale : vector<4xf32>
      %row_in_wave = index.add %frag_row_offset, %publish_row : index
      %row_local = index.add %wave_row, %row_in_wave : index
      %out_row = index.add %base_m, %row_local : index
      %writes = index.cmp ult, %out_row, %m_bounded : index
      scf.if %writes {{
        %bounded = index.assume %out_row [lt(%out_row, %m_bounded)] : index
        %a_scale_value = view.load %a_scale_view[%bounded] : view<[%m_bounded]xf32> -> f32
        %a_scale_vector = vector.splat %a_scale_value : vector<4xf32>
        %g = vector.mulf %gate_scaled0, %a_scale_vector : vector<4xf32>
        %u = vector.mulf %up_scaled0, %a_scale_vector : vector<4xf32>
        %neg_g = vector.subf %zero4, %g : vector<4xf32>
        %e = vector.expf<afn> %neg_g : vector<4xf32>
        %den = vector.addf %one4, %e : vector<4xf32>
        %sig = vector.divf %one4, %den : vector<4xf32>
        %silu = vector.mulf %g, %sig : vector<4xf32>
        %prod = vector.mulf %silu, %u : vector<4xf32>
        %narrow = vector.fptrunc %prod : vector<4xf32> to vector<4xf16>
        vector.store %narrow, %c_view[%bounded, %out_col] : vector<4xf16>, view<[%m_bounded]x[%n_half]xf16>
      }}
    }}
    kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)
  }}
"""
    K += "  kernel.return\n}\n"
    return K


if __name__ == "__main__":
    for bits in (4, 8):
        for mode in ("plain", "resid", "swiglu"):
            stem = f"gemm_i{bits}" + {"plain": "", "resid": "_resid", "swiglu": "_swiglu"}[mode] + "_256"
            (ROOT / "kernels" / f"{stem}.loom").write_text(generate(mode, bits=bits))
            print("wrote", stem)
