"""32-key transposed-product fp16 attention; requires transposed V.

Defaults: two query tiles, wide staging, native f16 probability repack,
and PV groups of two. This benchmark experiment fails full-trajectory quality
and is not selected by production builders.
"""
import os
from pathlib import Path

from gen_attention_query import generate as generate16, reduce_eight, ROOT


def generate():
    previous = dict(os.environ)
    try:
        os.environ.update(ATTN_STEM=os.environ.get("ATTN_STEM", "attention_query32"),
                          ATTN_QTILES=os.environ.get("ATTN_QTILES", "2"),
                          ATTN_VT="1", ATTN_SPLIT_STAGE="0", ATTN_QK_CHAINS="1",
                          ATTN_PACKED_REPACK="0", ATTN_TILE="16", ATTN_QLDS="1")
        stem, text = generate16()
    finally:
        os.environ.clear()
        os.environ.update(previous)
    hoist = int(os.environ.get("ATTN_HOIST", "4"))
    waves = 4 * int(os.environ.get("ATTN_QTILES", "2"))
    slots = 2 if os.environ.get("ATTN_DOUBLE_BUFFER", "0") == "1" else 1
    direct = os.environ.get("ATTN_DIRECT_OUT", "1") == "1"
    wide = os.environ.get("ATTN_WIDE_STAGE", "1") == "1"
    native_pack = os.environ.get("ATTN_NATIVE_PACK", "1") == "1"
    qrow = (8 - hoist) * 16 + 8 if hoist < 8 else 0
    qbytes = waves * 16 * qrow * 2

    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new)

    replace("  %c32 = index.constant 32 : index", "  %c31 = index.constant 31 : index\n  %c32 = index.constant 32 : index")
    replace("  %v_tile_offset = index.constant 4352 : offset", "  %v_tile_offset = index.constant 8704 : offset")
    replace(f"  %q_tile_offset = index.constant {10496 * slots} : offset", f"  %q_tile_offset = index.constant {18944 * slots} : offset")
    replace(f"  %lds_bytes = index.constant {max(10496 * slots + qbytes, 0 if direct else waves * 1024)} : offset",
            f"  %lds_bytes = index.constant {max(18944 * slots + qbytes, 0 if direct else waves * 1024)} : offset")
    if slots == 2:
        replace("    %slot_bytes = index.constant 10496 : offset", "    %slot_bytes = index.constant 18944 : offset")
    replace("%tile_origin_limit = index.sub %padded_tokens, %c16", "%tile_origin_limit = index.sub %padded_tokens, %c32")
    replace("%key_rounded = index.add %tokens0, %c15", "%key_rounded = index.add %tokens0, %c31")
    replace("%key_tile_count = index.div %key_rounded, %c16", "%key_tile_count = index.div %key_rounded, %c32")
    replace("%key_origin1 = index.mul %key_tile, %c16", "%key_origin1 = index.mul %key_tile, %c32")
    replace("mul(%key_origin1, 16)", "mul(%key_origin1, 32)")
    text = text.replace("view<16x136xf16>", "view<32x136xf16>").replace("view<128x24xf16>", "view<128x40xf16>")
    marker = "    vector.store %v_chunk, %v_tile[%workitem, %c0] : vector<16xf16>, view<128x40xf16>"
    replace(marker, marker + """
    %st_key_hi = index.add %st_key, %c16 : index
    %st_row_hi0 = index.add %st_row0, %c16 : index
    %st_row_hi = index.assume %st_row_hi0 [lt(%st_row_hi0, %padded_tokens)] : index
    %k_chunk_hi = vector.load %k_view[%st_row_hi, %st_col] : view<[%padded_tokens]x[%kv_stride0]xf16> -> vector<16xf16>
    vector.store %k_chunk_hi, %k_tile[%st_key_hi, %st_chunk] : vector<16xf16>, view<32x136xf16>
    %key_hi0 = index.add %key_origin0, %c16 : index
    %key_hi_limit = index.sub %padded_tokens, %c16 : index
    %key_hi = index.assume %key_hi0 [le(%key_hi0, %key_hi_limit), mul(%key_hi0, 16)] : index
    %v_chunk_hi = vector.load %v_view[%vchan, %key_hi] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>
    vector.store %v_chunk_hi, %v_tile[%workitem, %c16] : vector<16xf16>, view<128x40xf16>
""")
    start = text.index("    %k_data0 = vector.load")
    # With HOIST=0 the first Q load precedes the first K load.
    if hoist == 0:
        start = text.index("    %q_data0 = vector.load")
    end = text.index("    %scaled0 = vector.mulf", start)
    body = "    %lane_hi = index.add %lane_column, %c16 : index\n"
    for c in range(8):
        if c >= hoist:
            body += f"""    %q_data{c} = vector.load %q_tile[%lane_column, %c{(c - hoist) * 16}] : view<16x{qrow}xf16> -> vector<16xf16>
    %lhs{c} = vector.fragment<rhs> %q_data{c} shape [%k_frag, %n] : vector<16xf16>
"""
        for half, row in (("lo", "lane_column"), ("hi", "lane_hi")):
            acc = "%init" if c == 0 else f"%qk_{half}{c - 1}"
            body += f"""    %kd_{half}{c} = vector.load %k_tile[%{row}, %c{c * 16}] : view<32x136xf16> -> vector<16xf16>
    %kf_{half}{c} = vector.fragment<lhs> %kd_{half}{c} shape [%m, %k_frag] : vector<16xf16>
    %qk_{half}{c} = vector.mma %kf_{half}{c}, %lhs{c}, {acc} : vector<16xf16>, vector<16xf16>, vector<8xf32>
"""
    text = text[:start] + body + text[end:]
    start = text.index("    %scaled0 = vector.mulf")
    end = text.index("    scf.yield %next_max, %next_sum", start)
    body = """    %last_tile = index.sub %key_tile_count, %c1 : index
    %is_tail = index.cmp eq, %key_tile, %last_tile : index
"""
    for half, offset in (("lo", 0), ("hi", 16)):
        body += f"""    %scaled0_{half} = vector.mulf %qk_{half}7, %scale_vector : vector<8xf32>
    %scaled_{half} = scf.if %is_tail -> (vector<8xf32>) {{
    %half_origin_{half} = index.add %key_origin0, %c{offset} : index
"""
        for i in range(8):
            body += f"""    %score_{half}{i} = vector.extract %scaled0_{half}[{i}] : vector<8xf32> -> f32
    %rel_{half}{i} = index.add %lane_group, %cj{i * 2} : index
    %key_{half}{i} = index.add %half_origin_{half}, %rel_{half}{i} : index
    %valid_{half}{i} = index.cmp ult, %key_{half}{i}, %tokens0 : index
    %masked_{half}{i} = scf.select %valid_{half}{i}, %score_{half}{i}, %negative_large : f32
"""
        body += f"    %tail_{half} = vector.from_elements " + ", ".join(f"%masked_{half}{i}" for i in range(8)) + " : vector<8xf32>\n"
        body += f"""      scf.yield %tail_{half} : vector<8xf32>
    }} else {{
      scf.yield %scaled0_{half} : vector<8xf32>
    }}
"""
        body += reduce_eight(f"scaled_{half}", f"max_{half}", "maxnumf")
    body += """    %local_max = scalar.maxnumf %max_lo, %max_hi : f32
    %partner_max, %max_ok = kernel.subgroup.shuffle<xor> %local_max, %i32_16, %i32_32 : f32, i32, i32
    %tile_max = scalar.maxnumf %local_max, %partner_max : f32
    %next_max = scalar.maxnumf %row_max, %tile_max : f32
    %max_vector = vector.splat %next_max : vector<8xf32>
    %old_delta = scalar.subf %row_max, %next_max : f32
    %old_scale = scalar.expf<afn> %old_delta : f32
    %old_scale_vector = vector.splat %old_scale : vector<8xf32>
"""
    for half in ("lo", "hi"):
        body += f"""    %delta_{half} = vector.subf %scaled_{half}, %max_vector : vector<8xf32>
    %weight_{half} = vector.expf<afn> %delta_{half} : vector<8xf32>
"""
        body += reduce_eight(f"weight_{half}", f"sum_{half}", "addf")
        if native_pack:
            # gfx11's f16 result fragment keeps each value in the low half
            # of a register. Convert once before exchanging lanes; the high
            # halves are ignored by the native result-to-RHS transition.
            body += f"    %half_{half} = vector.fptrunc %weight_{half} : vector<8xf32> to vector<8xf16>\n"
            for i in range(8):
                body += f"    %ph_{half}{i} = vector.extract %half_{half}[{i}] : vector<8xf16> -> f16\n"
            body += f"    %native_{half} = vector.from_elements " + ", ".join(
                value for i in range(8) for value in (f"%ph_{half}{i}", "%zero_f16")) + " : vector<16xf16>\n"
            body += f"""    %wf_{half} = vector.fragment<result> %native_{half} shape [%m, %n] : vector<16xf16>
    %prob_{half} = vector.fragment.repack<rhs> %wf_{half} shape [%k_frag, %n] : vector<16xf16> -> vector<16xf16>
"""
        else:
            body += f"""    %wf_{half} = vector.fragment<result> %weight_{half} shape [%m, %n] : vector<8xf32>
    %prob_{half} = vector.fragment.repack<rhs> %wf_{half} shape [%k_frag, %n] : vector<8xf32> -> vector<16xf16>
"""
    body += """    %local_sum = scalar.addf %sum_lo, %sum_hi : f32
    %scaled_sum = scalar.mulf %row_sum, %old_scale : f32
    %next_sum = scalar.addf %scaled_sum, %local_sum : f32
"""
    group = int(os.environ.get("ATTN_PV_GROUP", "2"))
    for c in range(8):
        body += f"""    %vrow{c} = index.add %lane_column, %c{c * 16} : index
    %rescaled{c} = vector.mulf %acc{c}, %old_scale_vector : vector<8xf32>
"""
        for half, offset, acc, out in (("lo", 0, f"rescaled{c}", f"pv_lo{c}"), ("hi", 16, f"pv_lo{c}", f"next{c}")):
            body += f"""    %vd_{half}{c} = vector.load %v_tile[%vrow{c}, %c{offset}] : view<128x40xf16> -> vector<16xf16>
    %vf_{half}{c} = vector.fragment<lhs> %vd_{half}{c} shape [%m, %k_frag] : vector<16xf16>
    %{out} = vector.mma %vf_{half}{c}, %prob_{half}, %{acc} : vector<16xf16>, vector<16xf16>, vector<8xf32>
"""
        if group and (c + 1) % group == 0 and c != 7:
            body += "    kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)\n"
    if slots == 1:
        body += "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
    text = text[:start] + body + text[end:]
    if native_pack:
        replace("  %zero_f32 = scalar.constant 0.0 : f32",
                "  %zero_f16 = scalar.constant 0.0 : f16\n  %zero_f32 = scalar.constant 0.0 : f32")
    if wide:
        # 128 staging threads each move 32 contiguous halves from K and V.
        # This avoids the distant second K-row load and its base-address
        # updates while covering exactly the same 32x128 tiles.
        replace("%st_key = index.div %workitem, %c8", "%st_key = index.div %workitem, %c4")
        replace("%st_chunk0 = index.rem %workitem, %c8", "%st_chunk0 = index.rem %workitem, %c4")
        replace("%st_chunk = index.mul %st_chunk0, %c16", "%st_chunk = index.mul %st_chunk0, %c32")
        replace("%kv_col_limit = index.sub %kv_stride0, %c16", "%kv_col_limit = index.sub %kv_stride0, %c32")
        replace("mul(%st_col0, 16)", "mul(%st_col0, 32)")
        start = text.index("    %k_chunk = vector.load")
        end = text.index("    kernel.barrier<workgroup> scope(workgroup)", start)
        text = text[:start] + """    %k_chunk = vector.load %k_view[%st_row, %st_col] : view<[%padded_tokens]x[%kv_stride0]xf16> -> vector<32xf16>
    %vchan0 = index.add %kv_base0, %workitem : index
    %vchan = index.assume %vchan0 [lt(%vchan0, %kv_stride0)] : index
    %v_chunk = vector.load %v_view[%vchan, %key_origin0] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<32xf16>
    vector.store %k_chunk, %k_tile[%st_key, %st_chunk] : vector<32xf16>, view<32x136xf16>
    vector.store %v_chunk, %v_tile[%workitem, %c0] : vector<32xf16>, view<128x40xf16>
    }
""" + text[end:]
    return stem, text.replace("Experimental transposed-product", "Transposed-product").replace(
        "generated by tools/gen_attention_query.py", "generated by tools/gen_attention_query32.py")


if __name__ == "__main__":
    stem, source = generate()
    path = ROOT / ("kernels" if stem == "attention_query32" else "experiments") / f"{stem}.loom"
    path.write_text(source)
    print("wrote", path)
