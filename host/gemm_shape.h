#pragma once
#include <initializer_list>
// The int4 GEMM launch shape rules shared by the kernel builders and the
// session, mirrored by scripts/build_kernels.py. Both sides derive the same
// numbers; the bundle's launch metadata records them so a session rejects a
// bundle built by a different rule.
namespace krea2_shape {

// Operand row pitch in k elements. A row of 8192 bytes (K = 16384 int4) makes
// the rows a GEMM step touches alias in the cache; one extra k step of padding
// took the down projection from 61 to 77 TOPS. Rows of 3072 bytes (K = 6144)
// showed no such effect (plain 0.99x, swiglu 0.99x, wo 1.05x raw time for 2%
// more bytes), so they stay dense.
inline int gemm_pitch(int k) { return k % 8192 == 0 ? k + 128 : k; }

// m-tiles per raster group. The 256-row kernels shorten their last raster
// group in-kernel, so they always take the full group of 4. The 128-row
// kernels pad the grid: 1 for a single tile row, else of 4, 3, 2 the one that
// pads the tile rows least (ties to the larger group).
inline int gemm_m_group(int tokens, int rows) {
  int tiles = (tokens + rows - 1) / rows;
  if (rows == 256)
    return 4;
  if (tiles == 1)
    return 1;
  int best = 4;
  for (int candidate : {3, 2})
    if ((tiles + candidate - 1) / candidate * candidate <
        (tiles + best - 1) / best * best)
      best = candidate;
  return best;
}

// Launch grid rows: exactly the tiles for the shortening 256-row kernels, the
// tiles padded to whole raster groups for the 128-row kernels.
inline int gemm_grid_rows(int tokens, int rows, int m_group) {
  int tiles = (tokens + rows - 1) / rows;
  if (rows == 256)
    return tiles;
  return (tiles + m_group - 1) / m_group * m_group;
}

// Workgroup tile rows. The 256x128 tile runs 4-11% faster per row than the
// 128x128 tile (paired A/B at 2064..16896 tokens) but rounds M up to 256, so
// it is chosen when its rows are within 8% of the 128-row grid's padded rows
// and there are at least 2048 tokens (at 1040 tokens it lost 3%: five 256-row
// tiles against nine 128-row tiles).
constexpr int WIDE_TILE_TOKENS = 2048;
inline int gemm_rows(int tokens) {
  if (tokens < WIDE_TILE_TOKENS)
    return 128;
  int wide = (tokens + 255) / 256;
  int narrow = gemm_grid_rows(tokens, 128, gemm_m_group(tokens, 128));
  return 50 * wide <= 27 * narrow ? 256 : 128;
}

} // namespace krea2_shape
