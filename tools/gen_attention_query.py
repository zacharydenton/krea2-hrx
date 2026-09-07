"""Experimental fp16 attention with queries in the WMMA result columns.

Compute S^T = K Q^T and O^T = V^T P^T. On gfx11 a result lane owns one
column and eight even/odd rows. Softmax therefore reduces eight local keys
and exchanges one scalar with lane xor 16. The score fragment becomes the
PV RHS with a half-wave exchange, without a shared-memory transpose.

The existing generator supplies the tested GQA staging and output plumbing.
Variants stay in experiments/ until correctness and paired timing qualify them.
"""
import os
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent.parent


def reduce_eight(vector, name, operation):
    # vector.reduce currently lowers to an eight-instruction serial chain.
    # An explicit tree has only three dependent levels. Sum reassociation is
    # intentional; the independent fp16 accuracy checks cover its rounding.
    lines = [f"    %{name}_element{i} = vector.extract %{vector}[{i}] : vector<8xf32> -> f32"
             for i in range(8)]
    values = [f"%{name}_element{i}" for i in range(8)]
    level = 0
    while len(values) > 1:
        result = []
        for i in range(0, len(values), 2):
            out = f"%{name}" if len(values) == 2 else f"%{name}_{level}_{i // 2}"
            lines.append(f"    {out} = scalar.{operation} {values[i]}, {values[i + 1]} : f32")
            result.append(out)
        values = result
        level += 1
    return "\n".join(lines) + "\n"


def generate():
    stem = os.environ.get("ATTN_STEM", "attention_query_f16")
    assert stem != "attention_gqa_lds_f16_wmma", "keep the deployment baseline intact"
    assert int(os.environ.get("ATTN_TILE", "16")) == 16
    previous = dict(os.environ)
    try:
        os.environ.update(ATTN_NO_WRITE="1", ATTN_STEM=stem, ATTN_TILE="16")
        base = runpy.run_path(str(ROOT / "tools/gen_attention_lds.py"))
    finally:
        os.environ.clear()
        os.environ.update(previous)
    text = base["K"]
    hoist = base["HOIST"]
    assert hoist in (0, 2, 4, 6, 8)

    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new)

    replace("  %i32_32 = scalar.constant 32 : i32",
            "  %i32_16 = scalar.constant 16 : i32\n  %i32_32 = scalar.constant 32 : i32")
    replace("  %lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer",
            "  %query_row0 = index.add %query_origin0, %lane_column : index\n"
            "  %query_row = index.assume %query_row0 [lt(%query_row0, %padded_tokens)] : index\n"
            "  %lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer")
    # A raw Q row is the column of Q^T consumed as the RHS. A fragment load
    # with the RHS role would instead load a column of the original Q view.
    for c in range(hoist):
        replace(f"  %lhs{c} = vector.fragment.load<lhs> %q_view[%query_origin0, %q_channel{c}] shape [%m, %k_frag] : view<[%padded_tokens]x[%q_stride0]xf16> -> vector<16xf16>",
                f"  %q_hoist{c} = vector.load %q_view[%query_row, %q_channel{c}] : view<[%padded_tokens]x[%q_stride0]xf16> -> vector<16xf16>\n"
                f"  %lhs{c} = vector.fragment<rhs> %q_hoist{c} shape [%k_frag, %n] : vector<16xf16>")
    for c in range(hoist, 8):
        if base["QROW"]:
            replace(f"    %lhs{c} = vector.fragment<lhs> %q_data{c} shape [%m, %k_frag] : vector<16xf16>",
                    f"    %lhs{c} = vector.fragment<rhs> %q_data{c} shape [%k_frag, %n] : vector<16xf16>")
        else:
            replace(f"    %lhs{c} = vector.fragment.load<lhs> %q_view[%query_origin0, %q_channel{c}] shape [%m, %k_frag] : view<[%padded_tokens]x[%q_stride0]xf16> -> vector<16xf16>",
                    f"    %q_reload{c} = vector.load %q_view[%query_row, %q_channel{c}] : view<[%padded_tokens]x[%q_stride0]xf16> -> vector<16xf16>\n"
                    f"    %lhs{c} = vector.fragment<rhs> %q_reload{c} shape [%k_frag, %n] : vector<16xf16>")
    for c in range(8):
        replace(f"%rhs{c} = vector.fragment<rhs> %k_data{c} shape [%k_frag, %n]",
                f"%rhs{c} = vector.fragment<lhs> %k_data{c} shape [%m, %k_frag]")
        replace(f"vector.mma %lhs{c}, %rhs{c},", f"vector.mma %rhs{c}, %lhs{c},")
    qk_chains = int(os.environ.get("ATTN_QK_CHAINS", "1"))
    assert qk_chains in (1, 2, 4)
    if qk_chains > 1:
        for c in range(1, 8):
            accumulator = "%init" if c < qk_chains else f"%qk{c - qk_chains}"
            replace(f"vector.mma %rhs{c}, %lhs{c}, %qk{c - 1}",
                    f"vector.mma %rhs{c}, %lhs{c}, {accumulator}")
        line_start = text.index("    %raw_scores = vector.mma")
        line_end = text.index("\n", line_start)
        line = text[line_start:line_end].replace("%raw_scores", "%qk7")
        values = [f"%qk{c}" for c in range(8 - qk_chains, 8)]
        level = 0
        while len(values) > 1:
            next_values = []
            for i in range(0, len(values), 2):
                out = "%raw_scores" if len(values) == 2 else f"%qk_sum{level}_{i}"
                line += f"\n    {out} = vector.addf {values[i]}, {values[i + 1]} : vector<8xf32>"
                next_values.append(out)
            values = next_values
            level += 1
        text = text[:line_start] + line + text[line_end:]
    replace("%row_max = %negative_vector : vector<8xf32>, %row_sum = %zero_vector : vector<8xf32>",
            "%row_max = %negative_large : f32, %row_sum = %zero_f32 : f32")
    result_types = ", ".join(["f32", "f32"] + ["vector<8xf32>"] * 8)
    replace("-> (" + ", ".join(["vector<8xf32>"] * 10) + ") {", f"-> ({result_types}) {{")

    pv_group = int(os.environ.get("ATTN_PV_GROUP", "4"))
    tail_only = os.environ.get("ATTN_TAIL_ONLY", "1") == "1"
    direct_out = os.environ.get("ATTN_DIRECT_OUT", "1") == "1"
    transposed_v = os.environ.get("ATTN_VT", "0") == "1"
    double_buffer = os.environ.get("ATTN_DOUBLE_BUFFER", "0") == "1"
    packed_repack = os.environ.get("ATTN_PACKED_REPACK", "0") == "1"
    split_stage = os.environ.get("ATTN_SPLIT_STAGE", "0") == "1"
    assert pv_group in (0, 1, 2, 4, 8)
    # The RHS repack never uses the old per-wave probability scratch. Output
    # scratch (when requested) aliases K/V storage after the final barrier.
    kv_bytes = 16 * base["ROW"] * 2 + 128 * base["VROW"] * 2
    kv_storage = kv_bytes * (2 if double_buffer else 1)
    scratch_bytes = base["WAVES"] * base["SCRATCH"]
    q_bytes = base["WAVES"] * 16 * base["QROW"] * 2
    lds_bytes = max(kv_storage + q_bytes, 0 if direct_out else base["WAVES"] * 1024)
    replace(f"  %q_tile_offset = index.constant {kv_bytes + scratch_bytes} : offset",
            f"  %q_tile_offset = index.constant {kv_storage} : offset")
    replace(f"  %lds_bytes = index.constant {kv_bytes + scratch_bytes + q_bytes} : offset",
            f"  %lds_bytes = index.constant {lds_bytes} : offset")
    if transposed_v:
        replace("  %v_view = buffer.view %v_global[%c0_offset] : buffer -> view<[%padded_tokens]x[%kv_stride0]xf16>",
                "  %v_view = buffer.view %v_global[%c0_offset] : buffer -> view<[%kv_stride0]x[%padded_tokens]xf16>")
        replace("    %v_chunk = vector.load %v_view[%st_row_v, %st_col_v] : view<[%padded_tokens]x[%kv_stride0]xf16> -> vector<16xf16>",
                "    %vchan0 = index.add %kv_base0, %workitem : index\n"
                "    %vchan = index.assume %vchan0 [lt(%vchan0, %kv_stride0)] : index\n"
                "    %v_chunk = vector.load %v_view[%vchan, %key_origin0] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>")
        first = text.index("    %ve0 = vector.extract")
        last = text.index("\n", text.index("    view.store %ve15,", first))
        text = text[:first] + "    vector.store %v_chunk, %v_tile[%workitem, %c0] : vector<16xf16>, view<128x24xf16>" + text[last:]
    if split_stage:
        assert transposed_v and base["QTILES"] == 2
        first = text.index("    %st_row0 = index.add %key_origin0")
        last = text.index("    kernel.barrier<workgroup> scope(workgroup)", first)
        text = text[:first] + """    scf.if %st_active {
      %st_row0 = index.add %key_origin0, %st_key : index
      %st_row = index.assume %st_row0 [lt(%st_row0, %padded_tokens)] : index
      %k_chunk = vector.load %k_view[%st_row, %st_col] : view<[%padded_tokens]x[%kv_stride0]xf16> -> vector<16xf16>
      vector.store %k_chunk, %k_tile[%st_key, %st_chunk] : vector<16xf16>, view<16x136xf16>
    } else {
      %v_stage_row0 = index.sub %workitem, %c128 : index
      %v_stage_row = index.assume %v_stage_row0 [range(%v_stage_row0, 0, 127)] : index
      %vchan0 = index.add %kv_base0, %v_stage_row : index
      %vchan = index.assume %vchan0 [lt(%vchan0, %kv_stride0)] : index
      %v_chunk = vector.load %v_view[%vchan, %key_origin0] : view<[%kv_stride0]x[%padded_tokens]xf16> -> vector<16xf16>
      vector.store %v_chunk, %v_tile[%v_stage_row, %c0] : vector<16xf16>, view<128x24xf16>
    }
""" + text[last:]
    if double_buffer:
        replace("  %k_tile = buffer.view", "  %k_unused = buffer.view")
        replace("  %v_tile = buffer.view", "  %v_unused = buffer.view")
        marker = "    // stage K and V tiles (rows past the sequence are zero headroom)"
        replace(marker, f"""    %slot = index.rem %key_tile, %c2 : index
    %slot_bytes = index.constant {kv_bytes} : offset
    %slot_offset = index.scale %slot, %slot_bytes : index, offset -> offset
    %v_slot = index.add %slot_offset, %v_tile_offset : offset
    %k_tile = buffer.view %lds[%slot_offset] : buffer -> view<16x136xf16>
    %v_tile = buffer.view %lds[%v_slot] : buffer -> view<128x24xf16>
{marker}""")
    start = text.index("    %scaled0 = vector.mulf")
    end = text.index("  // Publish one 16x16 fragment", start)
    body = "    %scaled0 = vector.mulf %raw_scores, %scale_vector : vector<8xf32>\n"
    if tail_only:
        body += """    %last_tile = index.sub %key_tile_count, %c1 : index
    %is_tail = index.cmp eq, %key_tile, %last_tile : index
    %scaled = scf.if %is_tail -> (vector<8xf32>) {
"""
    for i in range(8):
        body += f"""    %score{i} = vector.extract %scaled0[{i}] : vector<8xf32> -> f32
    %key_relative{i} = index.add %lane_group, %cj{2 * i} : index
    %key_absolute{i} = index.add %key_origin0, %key_relative{i} : index
    %key_present{i} = index.cmp ult, %key_absolute{i}, %tokens0 : index
    %masked{i} = scf.select %key_present{i}, %score{i}, %negative_large : f32
"""
    body += f"    %{'tail_scaled' if tail_only else 'scaled'} = vector.from_elements " + ", ".join(f"%masked{i}" for i in range(8)) + " : vector<8xf32>\n"
    if tail_only:
        body += """      scf.yield %tail_scaled : vector<8xf32>
    } else {
      scf.yield %scaled0 : vector<8xf32>
    }
"""
    body += reduce_eight("scaled", "local_max", "maxnumf")
    body += """    %partner_max, %max_ok = kernel.subgroup.shuffle<xor> %local_max, %i32_16, %i32_32 : f32, i32, i32
    %tile_max = scalar.maxnumf %local_max, %partner_max : f32
    %next_max = scalar.maxnumf %row_max, %tile_max : f32
    %max_vector = vector.splat %next_max : vector<8xf32>
    %delta = vector.subf %scaled, %max_vector : vector<8xf32>
    %weight = vector.expf<afn> %delta : vector<8xf32>
    %old_delta = scalar.subf %row_max, %next_max : f32
    %old_scale = scalar.expf<afn> %old_delta : f32
    %old_scale_vector = vector.splat %old_scale : vector<8xf32>
"""
    body += reduce_eight("weight", "local_sum", "addf")
    body += """    %scaled_sum = scalar.mulf %row_sum, %old_scale : f32
    %next_sum = scalar.addf %scaled_sum, %local_sum : f32
"""
    if packed_repack:
        body += """    %weight_half = vector.fptrunc %weight : vector<8xf32> to vector<8xf16>
    %weight_bits = vector.bitcast %weight_half : vector<8xf16> to vector<4xi32>
    %partner_bits, %pack_ok = kernel.subgroup.shuffle<xor> %weight_bits, %i32_16, %i32_32 : vector<4xi32>, i32, i32
    %even_bits = scf.select %lane_group_even, %weight_bits, %partner_bits : vector<4xi32>
    %odd_bits = scf.select %lane_group_even, %partner_bits, %weight_bits : vector<4xi32>
    %even_half = vector.bitcast %even_bits : vector<4xi32> to vector<8xf16>
    %odd_half = vector.bitcast %odd_bits : vector<4xi32> to vector<8xf16>
"""
        for i in range(8):
            body += f"    %pe{i} = vector.extract %even_half[{i}] : vector<8xf16> -> f16\n"
            body += f"    %po{i} = vector.extract %odd_half[{i}] : vector<8xf16> -> f16\n"
        body += "    %interleaved_half = vector.from_elements " + ", ".join(f"%p{parity}{i}" for i in range(8) for parity in ("e", "o")) + " : vector<16xf16>\n"
        body += "    %probability = vector.fragment<rhs> %interleaved_half shape [%k_frag, %n] : vector<16xf16>\n"
    else:
        body += """
    %weight_fragment = vector.fragment<result> %weight shape [%m, %n] : vector<8xf32>
    %probability = vector.fragment.repack<rhs> %weight_fragment shape [%k_frag, %n] : vector<8xf32> -> vector<16xf16>
"""
    for c in range(8):
        body += f"""    %vrow{c} = index.add %lane_column, %c{16 * c} : index
    %v_data{c} = vector.load %v_tile[%vrow{c}, %c0] : view<128x24xf16> -> vector<16xf16>
    %v{c} = vector.fragment<lhs> %v_data{c} shape [%m, %k_frag] : vector<16xf16>
    %rescaled{c} = vector.mulf %acc{c}, %old_scale_vector : vector<8xf32>
    %next{c} = vector.mma %v{c}, %probability, %rescaled{c} : vector<16xf16>, vector<16xf16>, vector<8xf32>
"""
        if pv_group and (c + 1) % pv_group == 0 and c != 7:
            body += "    kernel.barrier<workgroup> scope(subgroup) ordering(acq_rel)\n"
    if not double_buffer:
        body += "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
    body += "    scf.yield %next_max, %next_sum, " + ", ".join(f"%next{c}" for c in range(8)) + f" : {result_types}\n  }}\n"
    if double_buffer:
        body += "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
    body += """  %partner_sum, %sum_ok = kernel.subgroup.shuffle<xor> %final_sum, %i32_16, %i32_32 : f32, i32, i32
  %total_sum = scalar.addf %final_sum, %partner_sum : f32
  %selected_sum = vector.splat %total_sum : vector<8xf32>
"""
    if direct_out:
        for c in range(8):
            body += f"""  %out{c} = vector.divf<nnan|ninf|nsz|arcp> %final{c}, %selected_sum : vector<8xf32>
  %out_fragment{c} = vector.fragment<result> %out{c} shape [%m, %n] : vector<8xf32>
  %out_halves{c} = vector.fragment.repack<rhs> %out_fragment{c} shape [%m, %n] : vector<8xf32> -> vector<16xf16>
"""
        # Finish all cross-lane operations before masking any stores. A single
        # publication region also avoids carrying its mask across repacks.
        body += """  %query_in_sequence = index.cmp ult, %query_row0, %tokens0 : index
  %query_in_output = index.cmp ult, %query_row0, %token_count : index
  %query_present = scalar.andi %query_in_sequence, %query_in_output : i1
  %publish = scalar.andi %query_present, %lane_group_even : i1
  scf.if %publish {
"""
        for c in range(8):
            body += f"""    %out_row{c} = index.assume %query_row0 [lt(%query_row0, %token_count)] : index
    %out_col{c}_0 = index.add %head_base0, %c{16 * c} : index
    %out_last{c} = index.sub %out_stride0, %c16 : index
    %out_col{c} = index.assume %out_col{c}_0 [le(%out_col{c}_0, %out_last{c}), mul(%out_col{c}_0, 16)] : index
    vector.store %out_halves{c}, %out_view[%out_row{c}, %out_col{c}] : vector<16xf16>, view<[%token_count]x[%out_stride0]xf16>
"""
        body += "  }\n"
        text = text[:start] + body + "  kernel.return\n}\n"
        text = "// Experimental transposed-product fp16 attention; generated by tools/gen_attention_query.py.\n" + text[text.index("amdgpu.target"):]
        return stem, text
    for c in range(8):
        body += f"  %out{c} = vector.divf<nnan|ninf|nsz|arcp> %final{c}, %selected_sum : vector<8xf32>\n"
    text = text[:start] + body + text[end:]
    # Transpose the result into the existing dense publication scratch.
    replace("  %result_view = buffer.view %lds[%wave_result_offset] : buffer -> view<16x16xf32>",
            "  %result_view = buffer.view %lds[%wave_result_offset] : buffer -> view<16x16xf32>\n"
            "  %transpose = encoding.layout.strided [1, 16] : encoding<layout>\n"
            "  %result_transposed = buffer.view %lds[%wave_result_offset] : buffer -> view<16x16xf32, %transpose>")
    replace("vector.fragment.store<result> %pick7, %result_view[%c0, %c0] shape [%m, %n] : vector<8xf32>, view<16x16xf32>",
            "vector.fragment.store<result> %pick7, %result_transposed[%c0, %c0] shape [%m, %n] : vector<8xf32>, view<16x16xf32, %transpose>")
    text = "// Experimental transposed-product fp16 attention; generated by tools/gen_attention_query.py.\n" + text[text.index("amdgpu.target"):]
    return stem, text


if __name__ == "__main__":
    stem, text = generate()
    out = ROOT / "experiments" / f"{stem}.loom"
    out.write_text(text)
    print("wrote", out)
