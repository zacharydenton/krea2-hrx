# NPU GEMM sources

`gemm.py`, `mm.cc` and `zero.cc` derive from AMD MLIR-AIE commit
`0d49a88b78240dc742ba505c6ba0e8d9957ce614`, under Apache-2.0 WITH LLVM-exception.
Original copyright and SPDX notices are retained; see LICENSE.txt.

The generator is restricted to XDNA2, Chess, native BF16 operands, column-major
B (the model's row-major transposed weights), and row-major F32 output. It uses
the checked-in tile sources. AIE API/compiler support files must be included in
the HRX toolchain inventory. BFP16 emulation is disabled. Chess is supplied separately by the user; it is
not redistributed with this crate. The local Chess intrinsic wrapper must match
the installed compiler (see dinov3-xdna2/scripts/setup_toolchain.sh).
