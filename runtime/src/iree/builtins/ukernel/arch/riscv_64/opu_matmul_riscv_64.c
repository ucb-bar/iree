// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// OPU matmul ukernel: reads packed inputs, writes 2D output directly.
//
// This is the OPU encoding resolver's ukernel. Unlike iree_uk_mmt4d, which
// writes to a packed 4D output buffer [M, N, M0, N0], this ukernel writes
// directly to a 2D output buffer [M*M0, N*N0] with a given row stride.
// This eliminates the need for a separate unpack dispatch.
//
// The compute is identical to the mmt4d OPU early handler:
//   - vle8.v loads from packed LHS/RHS
//   - VOPACC outer product accumulate into matrix registers m0-m3
//   - VMV_VR + vse32.v stores to 2D output at stride out_stride0
//
// The only difference is the output addressing:
//   mmt4d:     tile_out = base + i*out_stride + j*M0*N0;  row stride = N0
//   opu_matmul: sub_out = base + (i*M0+m)*out_stride0 + j*N0+n; row stride = out_stride0

#include "iree/builtins/ukernel/exported_bits.h"

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"

#if defined(IREE_UK_ARCH_RISCV_64)

// Store one 16x16 sub-tile from matrix register mx to 2D output.
// out_row: pointer to first element of this sub-tile's first row in 2D output.
// out_stride: full 2D row stride in i32 elements.
// nrows/ncols: how many rows/cols to store (up to 16).
#define OPU_STORE_SUBTILE_2D(mx, out_row, out_stride, nrows, ncols)            \
  do {                                                                         \
    for (int _r = 0; _r < (nrows); ++_r) {                                    \
      asm volatile("vsetvli zero, %2, e32, m4, ta, ma\n\t"                    \
                   ".insn r 0x57, 0x6, 0x5d, x0, %0, " #mx "\n\t"            \
                   "vse32.v v0, (%1)\n\t"                                      \
                   :                                                           \
                   : "r"(_r), "r"((out_row) + _r * (out_stride)),             \
                     "r"((iree_uk_index_t)(ncols))                             \
                   : "memory");                                                \
    }                                                                          \
  } while (0)

// Specialized 32x32 sub-block for full 2x2 tiles (all quadrants 16x16).
// Optimizations over the generic path:
//   1. No conditional branches — all 4 VOPACCs unconditional
//   2. Pointer increments instead of k * M0 * K0 multiply per iteration
//   3. Single asm block per K step — prevents compiler vsetvli insertion
//   4. Only 4 asm operands — avoids GPR spilling (8 operands caused spills)
//   5. Accumulate path loads all 4 quadrants (m0, m1, m2, m3)
static void iree_uk_opu_matmul_sub32x32_full(
    const iree_uk_int8_t* lhs_ptr, const iree_uk_int8_t* rhs_c0,
    const iree_uk_int8_t* rhs_c1, iree_uk_int32_t* sub_out,
    iree_uk_index_t out_stride0, iree_uk_int32_t K, int M0, int N0,
    bool accumulate) {
  const int HW = 16;

  // --- Init 4 accumulators to zero ---
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(HW));
  asm volatile("vmv.v.i v0, 0" : : : "memory");
  asm volatile(".insn r 0x57, 0x6, 0x59, x0, x0, x0" : : : "memory");
  asm volatile(".insn r 0x57, 0x6, 0x59, x1, x0, x0" : : : "memory");
  asm volatile(".insn r 0x57, 0x6, 0x59, x2, x0, x0" : : : "memory");
  asm volatile(".insn r 0x57, 0x6, 0x59, x3, x0, x0" : : : "memory");

  if (accumulate) {
    // Load existing 2D output into all 4 accumulators.
    // Rows 0..15 → m0 (cols 0..15), m1 (cols 16..31)
    for (int r = 0; r < HW; ++r) {
      iree_uk_int32_t* row = sub_out + r * out_stride0;
      asm volatile("vle32.v v0, (%0)" : : "r"(row) : "memory");
      asm volatile(".insn r 0x57, 0x6, 0x55, x0, %0, x0"
                   :
                   : "r"(r)
                   : "memory"); // VMV_RV m0
      asm volatile("vle32.v v0, (%0)" : : "r"(row + HW) : "memory");
      asm volatile(".insn r 0x57, 0x6, 0x55, x1, %0, x0"
                   :
                   : "r"(r)
                   : "memory"); // VMV_RV m1
    }
    // Rows 16..31 → m2 (cols 0..15), m3 (cols 16..31)
    for (int r = 0; r < HW; ++r) {
      iree_uk_int32_t* row = sub_out + (r + HW) * out_stride0;
      asm volatile("vle32.v v0, (%0)" : : "r"(row) : "memory");
      asm volatile(".insn r 0x57, 0x6, 0x55, x2, %0, x0"
                   :
                   : "r"(r)
                   : "memory"); // VMV_RV m2
      asm volatile("vle32.v v0, (%0)" : : "r"(row + HW) : "memory");
      asm volatile(".insn r 0x57, 0x6, 0x55, x3, %0, x0"
                   :
                   : "r"(r)
                   : "memory"); // VMV_RV m3
    }
  }

  // --- K loop: single asm block, pointer increments, no unroll ---
  // Only 4 operands to avoid GPR register spilling.
  const iree_uk_int8_t* lp = lhs_ptr;
  const iree_uk_int8_t* rp0 = rhs_c0;
  const iree_uk_int8_t* rp1 = rhs_c1;

  asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(HW));

  for (iree_uk_int32_t k = 0; k < K; ++k) {
    asm volatile(
        "vle8.v v16, (%0)\n\t"
        "vle8.v v17, (%1)\n\t"
        ".insn r 0x57, 0x2, 0x51, x0, x17, x16\n\t" // m0 += v16 x v17
        "vle8.v v18, (%2)\n\t"
        ".insn r 0x57, 0x2, 0x51, x1, x18, x16\n\t" // m1 += v16 x v18
        "vle8.v v19, (%3)\n\t"
        ".insn r 0x57, 0x2, 0x51, x2, x17, x19\n\t" // m2 += v19 x v17
        ".insn r 0x57, 0x2, 0x51, x3, x18, x19\n\t" // m3 += v19 x v18
        :
        : "r"(lp), "r"(rp0), "r"(rp1), "r"(lp + HW)
        : "memory");
    lp += M0;
    rp0 += N0;
    rp1 += N0;
  }

  // --- Store to 2D output ---
  // Single vsetvli for all stores. Pointer-increment pattern avoids
  // precomputing 16 lui row-offset constants, reducing register pressure
  // so the K-loop base pointers stay in registers (no stack spills).
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(HW));
  {
    // m0 (rows 0..15, cols 0..15) then m2 (rows 16..31, cols 0..15).
    iree_uk_int32_t* p = sub_out;
    for (int r = 0; r < HW; ++r) {
      asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x0\n\t"
                   "vse32.v v0, (%1)"
                   :
                   : "r"(r), "r"(p)
                   : "memory");
      p += out_stride0;
    }
    for (int r = 0; r < HW; ++r) {
      asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x2\n\t"
                   "vse32.v v0, (%1)"
                   :
                   : "r"(r), "r"(p)
                   : "memory");
      p += out_stride0;
    }
  }
  {
    // m1 (rows 0..15, cols 16..31) then m3 (rows 16..31, cols 16..31).
    iree_uk_int32_t* p = sub_out + HW;
    for (int r = 0; r < HW; ++r) {
      asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x1\n\t"
                   "vse32.v v0, (%1)"
                   :
                   : "r"(r), "r"(p)
                   : "memory");
      p += out_stride0;
    }
    for (int r = 0; r < HW; ++r) {
      asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x3\n\t"
                   "vse32.v v0, (%1)"
                   :
                   : "r"(r), "r"(p)
                   : "memory");
      p += out_stride0;
    }
  }
}

// Full OPU matmul loop nest with 2D output addressing.
//
// Parameters:
//   lhs_base:    packed LHS [M, K, M0, K0], row stride = lhs_stride0
//   rhs_base:    packed RHS [N, K, N0, K0], row stride = rhs_stride0
//   out_base:    2D output [M*M0, N*N0], row stride = out_stride0
//   M, N, K:     tile counts
//   M0, N0, K0:  tile sizes
//   out_stride0: 2D row stride in i32 elements (= N * N0 for contiguous 2D)
//   accumulate:  if true, add to existing output values
static void iree_uk_opu_matmul_loop(
    const iree_uk_int8_t* lhs_base, iree_uk_index_t lhs_stride0,
    const iree_uk_int8_t* rhs_base, iree_uk_index_t rhs_stride0,
    iree_uk_int32_t* out_base, iree_uk_index_t out_stride0,
    iree_uk_int32_t M, iree_uk_int32_t N, iree_uk_int32_t K,
    int M0, int N0, int K0, bool accumulate) {
  const int HW = 16; // OPU hardware width

  for (iree_uk_int32_t i = 0; i < M; ++i) {
    const iree_uk_int8_t* lhs_panel = lhs_base + i * lhs_stride0;

    for (iree_uk_int32_t j = 0; j < N; ++j) {
      const iree_uk_int8_t* rhs_panel = rhs_base + j * rhs_stride0;

      // Process M0 x N0 tile using 2x2 sub-tiling with m0-m3.
      for (int m_sub = 0; m_sub < M0; m_sub += 2 * HW) {
        int m_rem = M0 - m_sub;
        int m_hw0 = (m_rem < HW) ? m_rem : HW;
        int m_hw1 =
            (m_rem > HW) ? ((m_rem - HW < HW) ? m_rem - HW : HW) : 0;

        for (int n_sub = 0; n_sub < N0; n_sub += 2 * HW) {
          int n_rem = N0 - n_sub;

          // Fast path: full 2x2 sub-block (all quadrants 16x16), K0=1.
          if (m_rem >= 2 * HW && n_rem >= 2 * HW && K0 == 1) {
            iree_uk_int32_t* sub_out =
                out_base + (i * M0 + m_sub) * out_stride0 + j * N0 + n_sub;
            iree_uk_opu_matmul_sub32x32_full(
                lhs_panel + m_sub, rhs_panel + n_sub,
                rhs_panel + n_sub + HW, sub_out, out_stride0, K, M0, N0,
                accumulate);
            continue;
          }

          // Generic path for partial tiles.
          int n_hw0 = (n_rem < HW) ? n_rem : HW;
          int n_hw1 =
              (n_rem > HW) ? ((n_rem - HW < HW) ? n_rem - HW : HW) : 0;

          // --- Init 4 accumulators (m0, m1, m2, m3) ---
          // ALL vsetvli + vector ops fused into single asm blocks.
          // This file is compiled as LLVM bitcode where RISCVInsertVSETVLI
          // strips standalone asm volatile vsetvli instructions.
          asm volatile(
              "vsetvli zero, %0, e32, m4, ta, ma\n\t"
              "vmv.v.i v0, 0\n\t"
              ".insn r 0x57, 0x6, 0x59, x0, x0, x0\n\t"  // OPMVINBCAST m0
              ".insn r 0x57, 0x6, 0x59, x1, x0, x0\n\t"  // m1
              ".insn r 0x57, 0x6, 0x59, x2, x0, x0\n\t"  // m2
              ".insn r 0x57, 0x6, 0x59, x3, x0, x0\n\t"  // m3
              :
              : "r"((iree_uk_index_t)n_hw0)
              : "memory");

          if (accumulate) {
            // Load existing 2D output into accumulators.
            iree_uk_int32_t* sub_out =
                out_base + (i * M0 + m_sub) * out_stride0 + j * N0 + n_sub;
            for (int r = 0; r < m_hw0; ++r) {
              asm volatile(
                  "vsetvli zero, %2, e32, m4, ta, ma\n\t"
                  "vle32.v v0, (%0)\n\t"
                  ".insn r 0x57, 0x6, 0x55, x0, %1, x0\n\t"  // VMV_RV m0
                  :
                  : "r"(sub_out + r * out_stride0), "r"(r),
                    "r"((iree_uk_index_t)n_hw0)
                  : "memory");
            }
            if (m_hw1 > 0) {
              for (int r = 0; r < m_hw1; ++r) {
                asm volatile(
                    "vsetvli zero, %2, e32, m4, ta, ma\n\t"
                    "vle32.v v0, (%0)\n\t"
                    ".insn r 0x57, 0x6, 0x55, x2, %1, x0\n\t"  // VMV_RV m2
                    :
                    : "r"(sub_out + (r + HW) * out_stride0), "r"(r),
                      "r"((iree_uk_index_t)n_hw0)
                    : "memory");
              }
            }
          }

          // --- K loop: 2x2 VOPACC ---
          const iree_uk_int8_t* lhs_k = lhs_panel + m_sub;
          const iree_uk_int8_t* rhs_k0 = rhs_panel + n_sub;
          const iree_uk_int8_t* rhs_k1 = rhs_panel + n_sub + HW;

          // Pre-zero v16/v19 for narrow-M/N cases.
          asm volatile(
              "vsetvli zero, %0, e8, m1, ta, ma\n\t"
              "vmv.v.i v16, 0\n\t"
              "vmv.v.i v19, 0\n\t"
              :
              : "r"((iree_uk_index_t)HW)
              : "memory");

          // Branch: full tiles use original fast asm; narrow tiles use
          // fused vsetvli+vle blocks to prevent LLVM from stripping vsetvli.
          if (m_hw0 == HW && n_hw0 == HW) {
            // --- FAST PATH: full 16×16 sub-tile ---
            // NOTE: vsetvli MUST be inside the asm block because this file
            // is compiled as LLVM bitcode (-ffreestanding -O3) and LLVM's
            // RISCVInsertVSETVLI pass strips standalone asm volatile vsetvli.
            for (int k = 0; k < K; ++k) {
              const iree_uk_int8_t* lhs_kk = lhs_k + k * M0 * K0;
              const iree_uk_int8_t* rhs_kk0 = rhs_k0 + k * N0 * K0;
              const iree_uk_int8_t* rhs_kk1 = rhs_k1 + k * N0 * K0;
              for (int k0 = 0; k0 < K0; ++k0) {
                asm volatile("vsetvli zero, %2, e8, m1, ta, ma\n\t"
                             "vle8.v v16, (%0)\n\t"
                             "vle8.v v17, (%1)\n\t"
                             ".insn r 0x57, 0x2, 0x51, x0, x17, x16\n\t"
                             :
                             : "r"(lhs_kk + k0 * M0), "r"(rhs_kk0 + k0 * N0),
                               "r"((iree_uk_index_t)HW)
                             : "memory");
                if (n_hw1 > 0) {
                  asm volatile("vle8.v v18, (%0)\n\t"
                               ".insn r 0x57, 0x2, 0x51, x1, x18, x16\n\t"
                               :
                               : "r"(rhs_kk1 + k0 * N0)
                               : "memory");
                }
                if (m_hw1 > 0) {
                  asm volatile("vle8.v v19, (%0)\n\t"
                               ".insn r 0x57, 0x2, 0x51, x2, x17, x19\n\t"
                               :
                               : "r"(lhs_kk + k0 * M0 + HW)
                               : "memory");
                  if (n_hw1 > 0) {
                    asm volatile(
                        ".insn r 0x57, 0x2, 0x51, x3, x18, x19\n\t" : : :
                        "memory");
                  }
                }
              }
            }
          } else {
            // --- NARROW PATH: fused vsetvli to prevent LLVM stripping ---
            for (int k = 0; k < K; ++k) {
              const iree_uk_int8_t* lhs_kk = lhs_k + k * M0 * K0;
              const iree_uk_int8_t* rhs_kk0 = rhs_k0 + k * N0 * K0;
              const iree_uk_int8_t* rhs_kk1 = rhs_k1 + k * N0 * K0;
              for (int k0 = 0; k0 < K0; ++k0) {
                asm volatile(
                    "vsetvli zero, %2, e8, m1, tu, ma\n\t"
                    "vle8.v v16, (%0)\n\t"
                    "vsetvli zero, %3, e8, m1, tu, ma\n\t"
                    "vle8.v v17, (%1)\n\t"
                    "vsetvli zero, %4, e8, m1, ta, ma\n\t"
                    ".insn r 0x57, 0x2, 0x51, x0, x17, x16\n\t"
                    :
                    : "r"(lhs_kk + k0 * M0), "r"(rhs_kk0 + k0 * N0),
                      "r"((iree_uk_index_t)m_hw0), "r"((iree_uk_index_t)n_hw0), "r"((iree_uk_index_t)HW)
                    : "memory");
                if (n_hw1 > 0) {
                  asm volatile(
                      "vsetvli zero, %1, e8, m1, tu, ma\n\t"
                      "vle8.v v18, (%0)\n\t"
                      "vsetvli zero, %2, e8, m1, ta, ma\n\t"
                      ".insn r 0x57, 0x2, 0x51, x1, x18, x16\n\t"
                      :
                      : "r"(rhs_kk1 + k0 * N0),
                        "r"((iree_uk_index_t)n_hw1), "r"((iree_uk_index_t)HW)
                      : "memory");
                }
                if (m_hw1 > 0) {
                  asm volatile(
                      "vsetvli zero, %1, e8, m1, tu, ma\n\t"
                      "vle8.v v19, (%0)\n\t"
                      "vsetvli zero, %2, e8, m1, ta, ma\n\t"
                      ".insn r 0x57, 0x2, 0x51, x2, x17, x19\n\t"
                      :
                      : "r"(lhs_kk + k0 * M0 + HW),
                        "r"((iree_uk_index_t)m_hw1), "r"((iree_uk_index_t)HW)
                      : "memory");
                  if (n_hw1 > 0) {
                    asm volatile(
                        ".insn r 0x57, 0x2, 0x51, x3, x18, x19\n\t" : : :
                        "memory");
                  }
                }
              }
            }
          }

          // --- Store to 2D output ---
          // Row stride is out_stride0 (the full 2D row stride).
          iree_uk_int32_t* sub_out =
              out_base + (i * M0 + m_sub) * out_stride0 + j * N0 + n_sub;

          // m0: rows 0..m_hw0-1, cols 0..n_hw0-1
          OPU_STORE_SUBTILE_2D(x0, sub_out, out_stride0, m_hw0, n_hw0);

          // m2: rows HW..HW+m_hw1-1, cols 0..n_hw0-1
          if (m_hw1 > 0) {
            OPU_STORE_SUBTILE_2D(x2, sub_out + HW * out_stride0, out_stride0,
                                 m_hw1, n_hw0);
          }

          // m1: rows 0..m_hw0-1, cols HW..HW+n_hw1-1
          if (n_hw1 > 0) {
            OPU_STORE_SUBTILE_2D(x1, sub_out + HW, out_stride0, m_hw0, n_hw1);
          }

          // m3: rows HW..HW+m_hw1-1, cols HW..HW+n_hw1-1
          if (m_hw1 > 0 && n_hw1 > 0) {
            OPU_STORE_SUBTILE_2D(x3, sub_out + HW * out_stride0 + HW,
                                 out_stride0, m_hw1, n_hw1);
          }
        }
      }
    }
  }
}

// Exported OPU matmul ukernel entry point.
// Called from iree_codegen.ukernel.generic "iree_uk_opu_matmul".
//
// Arguments (after bufferization with strided_outer_dims=1):
//   lhs_buffer, lhs_offset, lhs_stride0
//   rhs_buffer, rhs_offset, rhs_stride0
//   out_buffer, out_offset, out_stride0
//   M, N, K       (tile counts, index type)
//   M0, N0, K0    (tile sizes, i32)
//   flags         (i32)
//   cpu_data      (pointer)
IREE_UK_EXPORT void iree_uk_opu_matmul(
    const void* lhs_buffer, iree_uk_index_t lhs_offset,
    iree_uk_index_t lhs_stride0, const void* rhs_buffer,
    iree_uk_index_t rhs_offset, iree_uk_index_t rhs_stride0,
    void* out_buffer, iree_uk_index_t out_offset,
    iree_uk_index_t out_stride0, iree_uk_index_t M, iree_uk_index_t N,
    iree_uk_index_t K, iree_uk_int32_t M0, iree_uk_int32_t N0,
    iree_uk_int32_t K0, iree_uk_uint32_t flags,
    const iree_uk_uint64_t* cpu_data) {
  // Trivial cases.
  if (M == 0 || N == 0) return;
  if (K == 0 && (flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)) return;

  const iree_uk_int8_t* lhs_base =
      (const iree_uk_int8_t*)lhs_buffer + lhs_offset;
  const iree_uk_int8_t* rhs_base =
      (const iree_uk_int8_t*)rhs_buffer + rhs_offset;
  iree_uk_int32_t* out_base = (iree_uk_int32_t*)out_buffer + out_offset;

  bool accumulate = flags & IREE_UK_FLAG_MMT4D_ACCUMULATE;

  iree_uk_opu_matmul_loop(lhs_base, lhs_stride0, rhs_base, rhs_stride0,
                           out_base, out_stride0, M, N, K, M0, N0, K0,
                           accumulate);
}

#endif // IREE_UK_ARCH_RISCV_64
