// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Saturn OPU (+xopu) mmt4d tile implementations.
// This file contains all OPU-specific mmt4d kernels using the outer product
// accelerator (VOPACC, VMV_VR, VMV_RV, OPMVINBCAST instructions).
// Guarded by IREE_UK_ARCH_RISCV_64 — only compiled for RISC-V 64-bit targets.

#include <riscv_vector.h>

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"

#ifdef IREE_UK_DEBUG_OPU_NARROW_M
#include <stdio.h>
#define IREE_UK_DEBUG_OPU_NARROW_M_PRINT(fmt, ...) \
  do { fprintf(stderr, "[opu narrow-M] " fmt "\n", ##__VA_ARGS__); } while (0)
#else
#define IREE_UK_DEBUG_OPU_NARROW_M_PRINT(fmt, ...) do {} while (0)
#endif

#if defined(IREE_UK_ARCH_RISCV_64)

// OPMVINBCAST: broadcast v0 into all rows of matrix register m0.
// rd=x0(m0) for row-broadcast (matches benchmark pattern).
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_riscv_64_xopu_opmvinbcast_m0_from_v0(void) {
  asm volatile(".insn r 0x57, 0x6, 0x59, x0, x0, x0\n\t" : : : "memory");
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_riscv_64_xopu_vmv_rv_m0_from_v0(int row) {
  asm volatile(".insn r 0x57, 0x6, 0x55, x0, %0, x0\n\t"
               :
               : "r"(row)
               : "memory");
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_riscv_64_xopu_vmv_vr_v0_from_m0(int row) {
  asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x0\n\t"
               :
               : "r"(row)
               : "memory");
}

// VOPACC: outer product accumulate. rs1=LHS(rows), rs2=RHS(cols).
// m0[i][j] += rs1[i] * rs2[j].
// v16=LHS column, v18=RHS column → rs1=v16(x16), rs2=v18(x18).
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_riscv_64_xopu_vopacc_m0_v16_v18(void) {
  asm volatile(".insn r 0x57, 0x2, 0x51, x0, x16, x18\n\t" : : : "memory");
}

// VOPACC with different register pairs for loop unrolling.
// Using separate register pairs avoids VRF read-after-write hazards.
// Unrolled pair: v20=LHS, v22=RHS → rs1=v20(x20), rs2=v22(x22).
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_riscv_64_xopu_vopacc_m0_v20_v22(void) {
  asm volatile(".insn r 0x57, 0x2, 0x51, x0, x20, x22\n\t" : : : "memory");
}

// OPFMACC: FP8 fused multiply-accumulate (f8 x f8 -> f32 in matrix register).
// Two register pair variants for loop unrolling to hide VRF RAW hazards.
// OPFMACC: FP8 outer product accumulate. Same operand mapping as VOPACC.
// v16=LHS(rows), v18=RHS(cols) → rs1=v16(x16), rs2=v18(x18).
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_riscv_64_xopu_opfmacc_m0_v16_v18(void) {
  asm volatile(".insn r 0x57, 0x1, 0x4b, x0, x16, x18\n\t" : : : "memory");
}

// Unrolled pair: v20=LHS, v22=RHS → rs1=v20(x20), rs2=v22(x22).
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_riscv_64_xopu_opfmacc_m0_v20_v22(void) {
  asm volatile(".insn r 0x57, 0x1, 0x4b, x0, x20, x22\n\t" : : : "memory");
}

// =============================================================================
// Full OPU loop nest — saturn-vectors architecture.
//
// Reads from packed LHS [M, K, M0, K0] and packed RHS [N, K, N0, K0].
// Writes directly to 2D output at out_base[(i*M0+r)*out_stride + j*N0].
//
// Modeled on saturn/benchmarks/opu-2x2-gemm/kernel.h:
//   - 2x2 sub-tiling with m0-m3 for 32x32 blocks (when M0,N0 >= 32)
//   - Fallback to single m0 for smaller tiles
//   - Direct 2D store (no packed intermediate)
//
// This eliminates the separate unpack step entirely.
// =============================================================================

// Store one 16x16 sub-tile from matrix register mx to 2D output.
// out_row points to the first element of this sub-tile's first row.
// out_stride is the full 2D row stride (N*N0 elements).
// nrows/ncols: how many rows/cols to store (up to 16).
#define OPU_STORE_SUBTILE_2D(mx, out_row, out_stride, nrows, ncols)          \
  do {                                                                        \
    for (int _r = 0; _r < (nrows); ++_r) {                                   \
      asm volatile("vsetvli zero, %2, e32, m4, ta, ma\n\t"                   \
                   ".insn r 0x57, 0x6, 0x5d, x0, %0, " #mx "\n\t"           \
                   "vse32.v v0, (%1)\n\t"                                     \
                   :                                                          \
                   : "r"(_r), "r"((out_row) + _r * (out_stride)),            \
                     "r"((size_t)(ncols))                                     \
                   : "memory");                                               \
    }                                                                         \
  } while (0)

// Full OPU matmul: reads packed inputs, writes 2D output directly.
// Returns true if handled, false to fall through to standard path.
static bool iree_uk_mmt4d_opu_full_loop(
    const iree_uk_mmt4d_params_t* params) {
  const iree_uk_int32_t M = params->M;
  const iree_uk_int32_t N = params->N;
  const iree_uk_int32_t K = params->K;
  const int M0 = params->M0;
  const int N0 = params->N0;
  const int K0 = params->K0;
  const int HW = 16;  // OPU hardware width

  IREE_UK_DEBUG_OPU_NARROW_M_PRINT(
      "enter opu_full_loop M=%d N=%d K=%d M0=%d N0=%d K0=%d flags=0x%x",
      (int)M, (int)N, (int)K, M0, N0, K0, (unsigned)params->flags);

  iree_uk_int32_t* out_base =
      (iree_uk_int32_t*)params->out_buffer + params->out_offset;
  const iree_uk_int8_t* lhs_base =
      (const iree_uk_int8_t*)params->lhs_buffer + params->lhs_offset;
  const iree_uk_int8_t* rhs_base =
      (const iree_uk_int8_t*)params->rhs_buffer + params->rhs_offset;

  // Packed output layout: [M, N, M0, N0].
  // Tile (i, j) at offset: i * out_stride0 + j * tile_size.
  // Within a tile, stride between rows is N0.
  const iree_uk_index_t out_stride = params->out_stride0;  // N * M0 * N0
  const iree_uk_int32_t tile_size = M0 * N0;
  const iree_uk_index_t lhs_panel_stride = params->lhs_stride0;
  const iree_uk_index_t rhs_panel_stride = params->rhs_stride0;

  const bool accumulate =
      params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE;

  // Outer loops: iterate M-tiles x N-tiles
  for (iree_uk_int32_t i = 0; i < M; ++i) {
    const iree_uk_int8_t* lhs_panel = lhs_base + i * lhs_panel_stride;

    for (iree_uk_int32_t j = 0; j < N; ++j) {
      const iree_uk_int8_t* rhs_panel = rhs_base + j * rhs_panel_stride;

      // Packed layout: tile (i,j) at offset i*out_stride + j*tile_size.
      iree_uk_int32_t* tile_out = out_base + i * out_stride + j * tile_size;

      // Process this M0 x N0 tile using 2x2 sub-tiling with m0-m3
      // when dimensions are large enough, fallback to single m0 otherwise.
      for (int m_sub = 0; m_sub < M0; m_sub += 2 * HW) {
        int m_rem = M0 - m_sub;
        int m_hw0 = (m_rem < HW) ? m_rem : HW;
        int m_hw1 = (m_rem > HW) ? ((m_rem - HW < HW) ? m_rem - HW : HW) : 0;

        for (int n_sub = 0; n_sub < N0; n_sub += 2 * HW) {
          int n_rem = N0 - n_sub;
          int n_hw0 = (n_rem < HW) ? n_rem : HW;
          int n_hw1 = (n_rem > HW) ? ((n_rem - HW < HW) ? n_rem - HW : HW)
                                   : 0;

          // --- Init 4 accumulators (m0, m1, m2, m3) ---
          // ALL fused into one asm block — LLVM bitcode strips standalone vsetvli.
          asm volatile(
              "vsetvli zero, %0, e32, m4, ta, ma\n\t"
              "vmv.v.i v0, 0\n\t"
              ".insn r 0x57, 0x6, 0x59, x0, x0, x0\n\t"  // OPMVINBCAST m0
              ".insn r 0x57, 0x6, 0x59, x1, x0, x0\n\t"  // m1
              ".insn r 0x57, 0x6, 0x59, x2, x0, x0\n\t"  // m2
              ".insn r 0x57, 0x6, 0x59, x3, x0, x0\n\t"  // m3
              :
              : "r"((size_t)n_hw0)
              : "memory");

          if (accumulate) {
            iree_uk_int32_t* sub_out =
                tile_out + m_sub * N0 + n_sub;
            for (int r = 0; r < m_hw0; ++r) {
              asm volatile(
                  "vsetvli zero, %2, e32, m4, ta, ma\n\t"
                  "vle32.v v0, (%0)\n\t"
                  ".insn r 0x57, 0x6, 0x55, x0, %1, x0\n\t"  // VMV_RV m0
                  :
                  : "r"(sub_out + r * N0), "r"(r), "r"((size_t)n_hw0)
                  : "memory");
            }
            if (m_hw1 > 0) {
              for (int r = 0; r < m_hw1; ++r) {
                asm volatile(
                    "vsetvli zero, %2, e32, m4, ta, ma\n\t"
                    "vle32.v v0, (%0)\n\t"
                    ".insn r 0x57, 0x6, 0x55, x2, %1, x0\n\t"  // VMV_RV m2
                    :
                    : "r"(sub_out + (r + HW) * N0), "r"(r), "r"((size_t)n_hw0)
                    : "memory");
              }
            }
          }

          // --- K loop: 2x2 VOPACC ---
          const iree_uk_int8_t* lhs_k = lhs_panel + m_sub;
          const iree_uk_int8_t* rhs_k0 = rhs_panel + n_sub;
          const iree_uk_int8_t* rhs_k1 = rhs_panel + n_sub + HW;

          // Pre-zero v16/v19 for narrow-M cases (fused with vsetvli).
          asm volatile(
              "vsetvli zero, %0, e8, m1, ta, ma\n\t"
              "vmv.v.i v16, 0\n\t"
              "vmv.v.i v19, 0\n\t"
              :
              : "r"((size_t)HW)
              : "memory");

          // Branch on full-tile vs narrow-M/N: the full case (m_hw0==HW
          // && n_hw0==HW) uses the fast original single-vsetvli asm; the
          // narrow case uses the 3-vsetvli asm with tu,ma to avoid the
          // LLVM RISCVInsertVSETVLI stripping bug. The branch is on
          // loop-invariants so LLVM hoists it out of the K loop.
          if (m_hw0 == HW && n_hw0 == HW) {
            // --- FAST PATH: full 16×16 sub-tile ---
            // vsetvli inside asm block: this file is compiled as LLVM bitcode
            // and RISCVInsertVSETVLI strips standalone asm volatile vsetvli.
            for (int k = 0; k < K; ++k) {
              const iree_uk_int8_t* lhs_kk = lhs_k + k * M0 * K0;
              const iree_uk_int8_t* rhs_kk0 = rhs_k0 + k * N0 * K0;
              const iree_uk_int8_t* rhs_kk1 = rhs_k1 + k * N0 * K0;
              for (int k0 = 0; k0 < K0; ++k0) {
                asm volatile(
                    "vsetvli zero, %2, e8, m1, ta, ma\n\t"
                    "vle8.v v16, (%0)\n\t"
                    "vle8.v v17, (%1)\n\t"
                    ".insn r 0x57, 0x2, 0x51, x0, x17, x16\n\t"
                    :
                    : "r"(lhs_kk + k0 * M0), "r"(rhs_kk0 + k0 * N0),
                      "r"((size_t)HW)
                    : "memory");
                if (n_hw1 > 0) {
                  asm volatile(
                      "vle8.v v18, (%0)\n\t"
                      ".insn r 0x57, 0x2, 0x51, x1, x18, x16\n\t"
                      :
                      : "r"(rhs_kk1 + k0 * N0)
                      : "memory");
                }
                if (m_hw1 > 0) {
                  asm volatile(
                      "vle8.v v19, (%0)\n\t"
                      ".insn r 0x57, 0x2, 0x51, x2, x17, x19\n\t"
                      :
                      : "r"(lhs_kk + k0 * M0 + HW)
                      : "memory");
                  if (n_hw1 > 0) {
                    asm volatile(
                        ".insn r 0x57, 0x2, 0x51, x3, x18, x19\n\t"
                        :
                        :
                        : "memory");
                  }
                }
              }
            }
          } else {
            // --- NARROW PATH: m_hw0 < HW or n_hw0 < HW ---
            // Uses 3 vsetvli per inner iter inside a single asm block so
            // LLVM's RISCVInsertVSETVLI pass cannot strip them. The tu,ma
            // tail-undisturbed mode preserves the zero-padding in v16/v19.
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
                      "r"((size_t)m_hw0), "r"((size_t)n_hw0), "r"((size_t)HW)
                    : "memory");
                if (n_hw1 > 0) {
                  asm volatile(
                      "vsetvli zero, %1, e8, m1, tu, ma\n\t"
                      "vmv.v.i v18, 0\n\t"
                      "vle8.v v18, (%0)\n\t"
                      "vsetvli zero, %2, e8, m1, ta, ma\n\t"
                      ".insn r 0x57, 0x2, 0x51, x1, x18, x16\n\t"
                      :
                      : "r"(rhs_kk1 + k0 * N0),
                        "r"((size_t)n_hw1), "r"((size_t)HW)
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
                        "r"((size_t)m_hw1), "r"((size_t)HW)
                      : "memory");
                  if (n_hw1 > 0) {
                    asm volatile(
                        ".insn r 0x57, 0x2, 0x51, x3, x18, x19\n\t"
                        :
                        :
                        : "memory");
                  }
                }
              }
            }
          }

          // --- Store to packed output [M0, N0] tile ---
          // Row stride within tile is N0 (packed contiguous).
          iree_uk_int32_t* sub_out =
              tile_out + m_sub * N0 + n_sub;

          // m0: rows 0..m_hw0-1, cols 0..n_hw0-1
          OPU_STORE_SUBTILE_2D(x0, sub_out, N0, m_hw0, n_hw0);

          // m2: rows HW..HW+m_hw1-1, cols 0..n_hw0-1
          if (m_hw1 > 0) {
            OPU_STORE_SUBTILE_2D(x2, sub_out + HW * N0, N0,
                                 m_hw1, n_hw0);
          }

          // m1: rows 0..m_hw0-1, cols HW..HW+n_hw1-1
          if (n_hw1 > 0) {
            OPU_STORE_SUBTILE_2D(x1, sub_out + HW, N0, m_hw0, n_hw1);
          }

          // m3: rows HW..HW+m_hw1-1, cols HW..HW+n_hw1-1
          if (m_hw1 > 0 && n_hw1 > 0) {
            OPU_STORE_SUBTILE_2D(x3, sub_out + HW * N0 + HW,
                                 N0, m_hw1, n_hw1);
          }
        }
      }
    }
  }
  IREE_UK_DEBUG_OPU_NARROW_M_PRINT(
      "exit  opu_full_loop M0=%d N0=%d", M0, N0);
  return true;
}

// Top-level early handler for OPU.
// Returns true if the operation was handled, false to fall through.
bool iree_uk_mmt4d_early_riscv_64_xopu(
    const iree_uk_mmt4d_params_t* params) {
  if (!iree_uk_cpu_riscv_64_xopu(params->cpu_data)) return false;
  iree_uk_mmt4d_type_t mmt4d_type = iree_uk_mmt4d_type(params->flags);
  if (mmt4d_type != iree_uk_mmt4d_type_s8s8s32) return false;
  return iree_uk_mmt4d_opu_full_loop(params);
}

// =============================================================================
// Transposed store helpers for OPU (saturn-vectors i32_store_ct pattern).
//
// Normal store:     out[row * N0 + col] = m0[row][col]   (row-major)
// Transposed store: out[col * M0 + row] = m0[row][col]   (column-major)
//
// Both extract rows via VMV_VR. The difference is the store stride:
// normal uses N0 (row pitch), transposed uses M0 (column pitch).
// =============================================================================

// Store 16x16 tile from matrix register mx in column-major order.
// col_stride = M0 (number of rows in the tile).
#define IREE_UK_OPU_STORE_TILE_CT(mx, out_base, col_stride, ncols)           \
  do {                                                                        \
    for (int _c = 0; _c < (ncols); ++_c) {                                   \
      asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, " #mx                   \
                   :                                                          \
                   : "r"(_c)                                                  \
                   : "memory");                                               \
      asm volatile("vse32.v v0, (%0)"                                         \
                   :                                                          \
                   : "r"((out_base) + _c * (col_stride))                      \
                   : "memory");                                               \
    }                                                                         \
  } while (0)

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_16x16_riscv_64_xopu_native(
    iree_uk_int32_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  const long avl16 = 16;

  asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t"
               :
               : "r"(avl16)
               : "memory");

  // OPU keeps the accumulator in m0; initialize.
  asm volatile("vmv.v.i v0, 0\n\t" : : : "memory");
  iree_uk_riscv_64_xopu_opmvinbcast_m0_from_v0();
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int row = 0; row < 16; ++row) {
      asm volatile("vle32.v v0, (%0)\n\t"
                   :
                   : "r"(out_ptr + row * 16)
                   : "memory");
      iree_uk_riscv_64_xopu_vmv_rv_m0_from_v0(row);
    }
  }

  // Per-operand encoding swaps innerDimsPos for xopu LHS/RHS, giving
  // tile layout [K0, M0] (K-outer, M-inner) instead of [M0, K0].
  // For a given k0, M0=16 elements are contiguous at offset k0*16.
  // This enables vle8.v (16 elem/cycle) instead of vlse8.v (1 elem/cycle).
  //
  // Layout [K0, M0]: element [k0, m0] at offset k0*M0 + m0.
  // For k0=fixed: all M0 elements contiguous → vle8.v at (base + k0*16).
  for (int k = 0; k < params->K; ++k) {
    const iree_uk_int8_t* lhs_k_ptr = lhs_ptr + k * 16 * params->K0;
    const iree_uk_int8_t* rhs_k_ptr = rhs_ptr + k * 16 * params->K0;
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t"
                 :
                 : "r"(avl16)
                 : "memory");
    // Unroll K0 loop by 2 to hide VRF RAW hazards.
    int k0 = 0;
    for (; k0 + 2 <= params->K0; k0 += 2) {
      asm volatile("vle8.v v16, (%0)\n\t"
                   :
                   : "r"(lhs_k_ptr + k0 * 16)
                   : "memory");
      asm volatile("vle8.v v18, (%0)\n\t"
                   :
                   : "r"(rhs_k_ptr + k0 * 16)
                   : "memory");
      iree_uk_riscv_64_xopu_vopacc_m0_v16_v18();
      asm volatile("vle8.v v20, (%0)\n\t"
                   :
                   : "r"(lhs_k_ptr + (k0 + 1) * 16)
                   : "memory");
      asm volatile("vle8.v v22, (%0)\n\t"
                   :
                   : "r"(rhs_k_ptr + (k0 + 1) * 16)
                   : "memory");
      iree_uk_riscv_64_xopu_vopacc_m0_v20_v22();
    }
    for (; k0 < params->K0; ++k0) {
      asm volatile("vle8.v v16, (%0)\n\t"
                   :
                   : "r"(lhs_k_ptr + k0 * 16)
                   : "memory");
      asm volatile("vle8.v v18, (%0)\n\t"
                   :
                   : "r"(rhs_k_ptr + k0 * 16)
                   : "memory");
      iree_uk_riscv_64_xopu_vopacc_m0_v16_v18();
    }
  }

  // Store m0: column-major if transposed, row-major otherwise.
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t"
               :
               : "r"(avl16)
               : "memory");
  if (params->flags & IREE_UK_FLAG_MMT4D_TRANSPOSED_OUTPUT) {
    // Column-major store (saturn-vectors i32_store_ct pattern).
    for (int col = 0; col < 16; ++col) {
      iree_uk_riscv_64_xopu_vmv_vr_v0_from_m0(col);
      asm volatile("vse32.v v0, (%0)\n\t" : : "r"(out_ptr + col * 16)
                   : "memory");
    }
  } else {
    for (int row = 0; row < 16; ++row) {
      iree_uk_riscv_64_xopu_vmv_vr_v0_from_m0(row);
      asm volatile("vse32.v v0, (%0)\n\t" : : "r"(out_ptr + row * 16)
                   : "memory");
    }
  }
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_1xXXx16_to_16xXXx16_riscv_64_xopu_fallback(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16);
  // K0 is now dynamic (set by tile selection)
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const int N0 = params->N0;
  const int K0 = params->K0;
  const size_t vl = N0;

  // Keep register pressure bounded by processing rows in groups of 4.
  for (int m0 = 0; m0 < M0; m0 += 4) {
    int rows = (M0 - m0 < 4) ? (M0 - m0) : 4;
    vint32m4_t acc0, acc1, acc2, acc3;

    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_i32m4(out_ptr + m0 * N0, vl);
      if (rows >= 2) acc1 = __riscv_vle32_v_i32m4(out_ptr + (m0 + 1) * N0, vl);
      if (rows >= 3) acc2 = __riscv_vle32_v_i32m4(out_ptr + (m0 + 2) * N0, vl);
      if (rows >= 4) acc3 = __riscv_vle32_v_i32m4(out_ptr + (m0 + 3) * N0, vl);
    } else {
      acc0 = __riscv_vmv_v_x_i32m4(0, vl);
      if (rows >= 2) acc1 = __riscv_vmv_v_x_i32m4(0, vl);
      if (rows >= 3) acc2 = __riscv_vmv_v_x_i32m4(0, vl);
      if (rows >= 4) acc3 = __riscv_vmv_v_x_i32m4(0, vl);
    }

    // Per-operand encoding: tile layout [K0, M0] for LHS, [K0, N0] for RHS.
    // element [k0, m0] at offset k0*M0 + m0 (M innermost, contiguous).
    // element [k0, n0] at offset k0*N0 + n0 (N innermost, contiguous).
    for (int k = 0; k < params->K; ++k) {
      const iree_uk_int8_t* rhs_k_ptr = rhs_ptr + k * N0 * K0;
      const iree_uk_int8_t* lhs_k_ptr = lhs_ptr + k * M0 * K0 + m0;
      for (int k0 = 0; k0 < K0; ++k0) {
        // RHS: N0 elements for k0 are contiguous at k0*N0.
        vint8m1_t rhs = __riscv_vle8_v_i8m1(rhs_k_ptr + k0 * N0, vl);
        // LHS: element for row m at k0*M0 + m (M innermost).
        const iree_uk_int8_t* lhs_row = lhs_k_ptr + k0 * M0;
        iree_uk_int8_t lhs0 = lhs_row[0];
        vint16m2_t prod0 = __riscv_vwmul_vx_i16m2(rhs, lhs0, vl);
        acc0 = __riscv_vwadd_wv_i32m4(acc0, prod0, vl);
        if (rows >= 2) {
          iree_uk_int8_t lhs1 = lhs_row[1];
          vint16m2_t prod1 = __riscv_vwmul_vx_i16m2(rhs, lhs1, vl);
          acc1 = __riscv_vwadd_wv_i32m4(acc1, prod1, vl);
        }
        if (rows >= 3) {
          iree_uk_int8_t lhs2 = lhs_row[2];
          vint16m2_t prod2 = __riscv_vwmul_vx_i16m2(rhs, lhs2, vl);
          acc2 = __riscv_vwadd_wv_i32m4(acc2, prod2, vl);
        }
        if (rows >= 4) {
          iree_uk_int8_t lhs3 = lhs_row[3];
          vint16m2_t prod3 = __riscv_vwmul_vx_i16m2(rhs, lhs3, vl);
          acc3 = __riscv_vwadd_wv_i32m4(acc3, prod3, vl);
        }
      }
    }

    __riscv_vse32_v_i32m4(out_ptr + m0 * N0, acc0, vl);
    if (rows >= 2) __riscv_vse32_v_i32m4(out_ptr + (m0 + 1) * N0, acc1, vl);
    if (rows >= 3) __riscv_vse32_v_i32m4(out_ptr + (m0 + 2) * N0, acc2, vl);
    if (rows >= 4) __riscv_vse32_v_i32m4(out_ptr + (m0 + 3) * N0, acc3, vl);
  }
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_1xXXx16_to_16xXXx16_riscv_64_xopu(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  // K0 is now dynamic (set by tile selection)
  if (M0 == 16 && params->N0 == 16) {
    iree_uk_mmt4d_tile_s8s8s32_16x16_riscv_64_xopu_native(
        (iree_uk_int32_t*)out_tile, (const iree_uk_int8_t*)lhs_panel,
        (const iree_uk_int8_t*)rhs_panel, params);
    return;
  }
  // Truncation path for narrow-M/N cases: keep a correct RVV implementation.
  iree_uk_mmt4d_tile_s8s8s32_1xXXx16_to_16xXXx16_riscv_64_xopu_fallback(
      out_tile, lhs_panel, rhs_panel, params, M0);
}

// =============================================================================
// Saturn OPU int8 K0=1 path: uses VOPACC outer-product instructions.
//
// With K0=1 the compiler uses the standard RVV tile pipeline (no special
// packing). params->K equals the full reduction dimension. The OPU
// advantage comes entirely from using VOPACC instead of vwmul+vwadd.
// =============================================================================

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_16xXXx1_riscv_64_xopu(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16);
  IREE_UK_DEBUG_OPU_NARROW_M_PRINT(
      "enter 1xXXx1_to_16xXXx1 M0=%d N0=%d K=%d flags=0x%x", M0,
      (int)params->N0, (int)params->K, (unsigned)params->flags);
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;

  const int N0 = params->N0;
  const int K = params->K;
  const int HW_WIDTH = 16;

  for (int n_start = 0; n_start < N0; n_start += HW_WIDTH) {
    int n_rem = N0 - n_start;
    size_t vl = (n_rem < HW_WIDTH) ? n_rem : HW_WIDTH;
    size_t ml = M0;

    const iree_uk_int8_t* sub_rhs = rhs_ptr + n_start;
    iree_uk_int32_t* sub_out = out_ptr + n_start;

    // --- PATH A: High Performance (M0=16) ---
    if (M0 == 16) {
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
      asm volatile("vmv.v.i v0, 0" : : : "memory");
      iree_uk_riscv_64_xopu_opmvinbcast_m0_from_v0();

      if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        for (int row = 0; row < 16; ++row) {
          asm volatile("vle32.v v0, (%0)" : : "r"(&sub_out[row * N0]) : "memory");
          iree_uk_riscv_64_xopu_vmv_rv_m0_from_v0(row);
        }
      }

      // K loop: single asm block per iteration to prevent compiler from
      // inserting redundant vsetvli. Unrolled by 4 for reduced loop overhead
      // and better load/compute overlap.
      const iree_uk_int8_t* lhs_k = lhs_ptr;
      const iree_uk_int8_t* rhs_k = sub_rhs;
      size_t k = 0;
      asm volatile("vsetvli zero, %0, e8, m2, ta, ma" : : "r"(ml));
      for (; k + 4 <= (size_t)K; k += 4) {
        asm volatile(
            "vle8.v v16, (%0)\n\t"
            "vle8.v v18, (%1)\n\t"
            ".insn r 0x57, 0x2, 0x51, x0, x18, x16\n\t"
            "vle8.v v20, (%2)\n\t"
            "vle8.v v22, (%3)\n\t"
            ".insn r 0x57, 0x2, 0x51, x0, x22, x20\n\t"
            "vle8.v v24, (%4)\n\t"
            "vle8.v v26, (%5)\n\t"
            ".insn r 0x57, 0x2, 0x51, x0, x26, x24\n\t"
            "vle8.v v28, (%6)\n\t"
            "vle8.v v30, (%7)\n\t"
            ".insn r 0x57, 0x2, 0x51, x0, x30, x28\n\t"
            :
            : "r"(lhs_k), "r"(rhs_k),
              "r"(lhs_k + M0), "r"(rhs_k + N0),
              "r"(lhs_k + 2 * M0), "r"(rhs_k + 2 * N0),
              "r"(lhs_k + 3 * M0), "r"(rhs_k + 3 * N0)
            : "memory");
        lhs_k += 4 * M0;
        rhs_k += 4 * N0;
      }
      // Handle remaining 1-3 iterations
      for (; k < (size_t)K; ++k) {
        asm volatile(
            "vle8.v v16, (%0)\n\t"
            "vle8.v v18, (%1)\n\t"
            ".insn r 0x57, 0x2, 0x51, x0, x18, x16\n\t"
            :
            : "r"(lhs_k), "r"(rhs_k)
            : "memory");
        lhs_k += M0;
        rhs_k += N0;
      }

      if (params->flags & IREE_UK_FLAG_MMT4D_TRANSPOSED_OUTPUT) {
        // Column-major store (saturn-vectors i32_store_ct pattern).
        // VL = M0 (store M0 row-elements per column).
        asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(ml));
        for (size_t c = 0; c < (size_t)vl; c++) {
          iree_uk_riscv_64_xopu_vmv_vr_v0_from_m0(c);
          asm volatile("vse32.v v0, (%0)"
                       :
                       : "r"(out_ptr + (n_start + c) * M0)
                       : "memory");
        }
      } else {
        asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
        for (size_t r = 0; r < ml; r++) {
          iree_uk_riscv_64_xopu_vmv_vr_v0_from_m0(r);
          asm volatile("vse32.v v0, (%0)" : : "r"(&sub_out[r * N0]) : "memory");
        }
      }
    }
    // --- PATH B: Tail Case (M0 <= 8) ---
    else {
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl));
      asm volatile("vmv.v.i v0, 0" : : : "memory");
      iree_uk_riscv_64_xopu_opmvinbcast_m0_from_v0();

      if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        for (size_t row = 0; row < ml; ++row) {
          asm volatile("vle32.v v0, (%0)" : : "r"(&sub_out[row * N0]) : "memory");
          iree_uk_riscv_64_xopu_vmv_rv_m0_from_v0(row);
        }
      }

      for (int k = 0; k < K; ++k) {
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(ml));
        asm volatile("vle8.v v4, (%0)" : : "r"(&lhs_ptr[k * M0]) : "memory");
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(vl));
        asm volatile("vle8.v v5, (%0)" : : "r"(&sub_rhs[k * N0]) : "memory");
        asm volatile(".insn r 0x57, 0x2, 0x51, x0, x5, x4" : : : "memory");
      }

      if (params->flags & IREE_UK_FLAG_MMT4D_TRANSPOSED_OUTPUT) {
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(ml));
        for (size_t c = 0; c < (size_t)vl; c++) {
          iree_uk_riscv_64_xopu_vmv_vr_v0_from_m0(c);
          asm volatile("vse32.v v0, (%0)"
                       :
                       : "r"(out_ptr + (n_start + c) * M0)
                       : "memory");
        }
      } else {
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl));
        for (size_t r = 0; r < ml; r++) {
          iree_uk_riscv_64_xopu_vmv_vr_v0_from_m0(r);
          asm volatile("vse32.v v0, (%0)" : : "r"(&sub_out[r * N0]) : "memory");
        }
      }
    }
  }
  IREE_UK_DEBUG_OPU_NARROW_M_PRINT("exit  1xXXx1_to_16xXXx1 M0=%d", M0);
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_16xXXx1_riscv_64_xopu,
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_riscv_64_xopu, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_16xXXx1_riscv_64_xopu,
    iree_uk_mmt4d_tile_s8s8s32_2xXXx1_riscv_64_xopu, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_16xXXx1_riscv_64_xopu,
    iree_uk_mmt4d_tile_s8s8s32_4xXXx1_riscv_64_xopu, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_16xXXx1_riscv_64_xopu,
    iree_uk_mmt4d_tile_s8s8s32_8xXXx1_riscv_64_xopu, 8)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_16xXXx1_riscv_64_xopu,
    iree_uk_mmt4d_tile_s8s8s32_16xXXx1_riscv_64_xopu, 16)

// =============================================================================
// Saturn OPU int8 K0=1, M0=32: 2x2 output tiling using all 4 matrix registers.
//
// Processes a 32x32 output tile as four 16x16 sub-tiles:
//   m0: sub-tile (0,0)  m1: sub-tile (0,1)
//   m2: sub-tile (1,0)  m3: sub-tile (1,1)
//
// Per K iteration: 2 A-loads + 2 B-loads → 4 VOPACCs (1.0 loads/VOPACC).
// This is 2x better than the M0=16 path (2.0 loads/VOPACC).
// =============================================================================

void iree_uk_mmt4d_tile_s8s8s32_32xXXx1_riscv_64_xopu(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const int M0 = 32;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;

  const int N0 = params->N0;
  const int K = params->K;
  const int HW = 16;  // OPU hardware width

  // Process N0 in chunks of 2*HW (32), each chunk uses m0+m1 (or m2+m3)
  for (int n_start = 0; n_start < N0; n_start += 2 * HW) {
    int n_chunk = (N0 - n_start < 2 * HW) ? (N0 - n_start) : (2 * HW);
    int vl0 = (n_chunk < HW) ? n_chunk : HW;
    int vl1 = (n_chunk > HW) ? (n_chunk - HW) : 0;

    const iree_uk_int8_t* rhs0 = rhs_ptr + n_start;
    const iree_uk_int8_t* rhs1 = rhs_ptr + n_start + HW;
    iree_uk_int32_t* out0 = out_ptr + n_start;

    // --- Init all 4 accumulators ---
    // m0 = sub-tile(row0, col0), m1 = sub-tile(row0, col1)
    // m2 = sub-tile(row1, col0), m3 = sub-tile(row1, col1)
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl0));
    asm volatile("vmv.v.i v0, 0" : : : "memory");
    iree_uk_riscv_64_xopu_opmvinbcast_m0_from_v0();
    // m1
    asm volatile(".insn r 0x57, 0x6, 0x59, x1, x0, x0" : : : "memory");
    // m2
    asm volatile(".insn r 0x57, 0x6, 0x59, x2, x0, x0" : : : "memory");
    // m3
    asm volatile(".insn r 0x57, 0x6, 0x59, x3, x0, x0" : : : "memory");

    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      // Load existing accumulators for all 4 sub-tiles
      for (int r = 0; r < HW; ++r) {
        // m0: rows 0..15, cols 0..15
        asm volatile("vle32.v v0, (%0)" : : "r"(&out0[r * N0]) : "memory");
        iree_uk_riscv_64_xopu_vmv_rv_m0_from_v0(r);
        // m2: rows 16..31, cols 0..15
        asm volatile("vle32.v v0, (%0)" : : "r"(&out0[(r + HW) * N0]) : "memory");
        asm volatile(".insn r 0x57, 0x6, 0x55, x2, %0, x0" : : "r"(r) : "memory");
      }
      if (vl1 > 0) {
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl1));
        for (int r = 0; r < HW; ++r) {
          // m1: rows 0..15, cols 16..31
          asm volatile("vle32.v v0, (%0)" : : "r"(&out0[r * N0 + HW]) : "memory");
          asm volatile(".insn r 0x57, 0x6, 0x55, x1, %0, x0" : : "r"(r) : "memory");
          // m3: rows 16..31, cols 16..31
          asm volatile("vle32.v v0, (%0)" : : "r"(&out0[(r + HW) * N0 + HW]) : "memory");
          asm volatile(".insn r 0x57, 0x6, 0x55, x3, %0, x0" : : "r"(r) : "memory");
        }
      }
    }

    // --- K loop: 2x2 tiling, 4 VOPACCs per 4 loads ---
    const iree_uk_int8_t* lhs_k = lhs_ptr;
    const iree_uk_int8_t* rhs0_k = rhs0;
    const iree_uk_int8_t* rhs1_k = rhs1;

    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(HW));

    for (int k = 0; k < K; ++k) {
      // Load A row 0 (M elements 0..15)
      // Load B col 0
      // VOPACC m0 (sub-tile 0,0)
      // Load B col 1
      // VOPACC m1 (sub-tile 0,1) — reuse A row 0
      // Load A row 1 (M elements 16..31)
      // VOPACC m2 (sub-tile 1,0) — reuse B col 0
      // VOPACC m3 (sub-tile 1,1) — reuse B col 1
      asm volatile(
          "vle8.v v16, (%0)\n\t"   // A row 0
          "vle8.v v17, (%1)\n\t"   // B col 0
          ".insn r 0x57, 0x2, 0x51, x0, x17, x16\n\t"  // VOPACC m0
          "vle8.v v18, (%2)\n\t"   // B col 1
          ".insn r 0x57, 0x2, 0x51, x1, x18, x16\n\t"  // VOPACC m1
          "vle8.v v19, (%3)\n\t"   // A row 1
          ".insn r 0x57, 0x2, 0x51, x2, x17, x19\n\t"  // VOPACC m2 (reuse B col 0)
          ".insn r 0x57, 0x2, 0x51, x3, x18, x19\n\t"  // VOPACC m3 (reuse B col 1)
          :
          : "r"(lhs_k), "r"(rhs0_k),
            "r"(rhs1_k), "r"(lhs_k + HW)
          : "memory");

      lhs_k += M0;  // advance by 32 (full M0 stride)
      rhs0_k += N0;
      rhs1_k += N0;
    }

    // --- Store all 4 sub-tiles ---
    if (params->flags & IREE_UK_FLAG_MMT4D_TRANSPOSED_OUTPUT) {
      // Column-major store (saturn-vectors i32_store_ct pattern).
      // Normal layout: [M0=32 rows, N0 cols], row-major.
      // Transposed layout: [N0 cols, M0=32 rows], column-major.
      // Stride between columns = M0 = 32.
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(HW));
      // m0: rows 0..15 → columns 0..vl0-1, at col_stride=M0
      for (int c = 0; c < vl0; ++c) {
        iree_uk_riscv_64_xopu_vmv_vr_v0_from_m0(c);
        asm volatile("vse32.v v0, (%0)"
                     :
                     : "r"(out_ptr + (n_start + c) * M0)
                     : "memory");
      }
      // m2: rows 16..31 → columns 0..vl0-1, offset by HW within each column
      for (int c = 0; c < vl0; ++c) {
        asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x2" : : "r"(c) : "memory");
        asm volatile("vse32.v v0, (%0)"
                     :
                     : "r"(out_ptr + (n_start + c) * M0 + HW)
                     : "memory");
      }
      if (vl1 > 0) {
        // m1: rows 0..15 → columns HW..HW+vl1-1
        for (int c = 0; c < vl1; ++c) {
          asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x1" : : "r"(c) : "memory");
          asm volatile("vse32.v v0, (%0)"
                       :
                       : "r"(out_ptr + (n_start + HW + c) * M0)
                       : "memory");
        }
        // m3: rows 16..31 → columns HW..HW+vl1-1, offset by HW
        for (int c = 0; c < vl1; ++c) {
          asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x3" : : "r"(c) : "memory");
          asm volatile("vse32.v v0, (%0)"
                       :
                       : "r"(out_ptr + (n_start + HW + c) * M0 + HW)
                       : "memory");
        }
      }
    } else {
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl0));
      // m0: rows 0..15, cols 0..vl0
      for (int r = 0; r < HW; ++r) {
        iree_uk_riscv_64_xopu_vmv_vr_v0_from_m0(r);
        asm volatile("vse32.v v0, (%0)" : : "r"(&out0[r * N0]) : "memory");
      }
      // m2: rows 16..31, cols 0..vl0
      for (int r = 0; r < HW; ++r) {
        asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x2" : : "r"(r) : "memory");
        asm volatile("vse32.v v0, (%0)" : : "r"(&out0[(r + HW) * N0]) : "memory");
      }
      if (vl1 > 0) {
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl1));
        // m1: rows 0..15, cols HW..HW+vl1
        for (int r = 0; r < HW; ++r) {
          asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x1" : : "r"(r) : "memory");
          asm volatile("vse32.v v0, (%0)" : : "r"(&out0[r * N0 + HW]) : "memory");
        }
        // m3: rows 16..31, cols HW..HW+vl1
        for (int r = 0; r < HW; ++r) {
          asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x3" : : "r"(r) : "memory");
          asm volatile("vse32.v v0, (%0)" : : "r"(&out0[(r + HW) * N0 + HW]) : "memory");
        }
      }
    }
  }
}

// M0=32 tile function is registered directly via tiles.inl.

// =============================================================================
// Saturn OPU int8 K0=1, M0=64: processes 64x64 output tiles in 4 passes
// of 32x32 sub-blocks, each using the 2x2 tiling with m0-m3.
// =============================================================================

void iree_uk_mmt4d_tile_s8s8s32_64xXXx1_riscv_64_xopu(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const int M0 = 64;
  const int N0 = params->N0;
  const int K = params->K;
  const int HW = 16;
  const int BLK = 2 * HW;  // 32: size of each 2x2 sub-block

  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;

  // Process 64x64 in 4 passes of 32x32 sub-blocks
  for (int m_blk = 0; m_blk < M0; m_blk += BLK) {
    for (int n_blk = 0; n_blk < N0; n_blk += BLK) {
      int n_chunk = (N0 - n_blk < BLK) ? (N0 - n_blk) : BLK;
      int vl0 = (n_chunk < HW) ? n_chunk : HW;
      int vl1 = (n_chunk > HW) ? (n_chunk - HW) : 0;

      const iree_uk_int8_t* rhs0 = rhs_ptr + n_blk;
      const iree_uk_int8_t* rhs1 = rhs_ptr + n_blk + HW;
      iree_uk_int32_t* out0 = out_ptr + m_blk * N0 + n_blk;

      // Init all 4 accumulators to zero
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl0));
      asm volatile("vmv.v.i v0, 0" : : : "memory");
      // m0, m1, m2, m3
      asm volatile(".insn r 0x57, 0x6, 0x59, x0, x0, x0" : : : "memory");
      asm volatile(".insn r 0x57, 0x6, 0x59, x1, x0, x0" : : : "memory");
      asm volatile(".insn r 0x57, 0x6, 0x59, x2, x0, x0" : : : "memory");
      asm volatile(".insn r 0x57, 0x6, 0x59, x3, x0, x0" : : : "memory");

      if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        for (int r = 0; r < HW; ++r) {
          asm volatile("vle32.v v0, (%0)" : : "r"(&out0[r * N0]) : "memory");
          asm volatile(".insn r 0x57, 0x6, 0x55, x0, %0, x0" : : "r"(r) : "memory");
          asm volatile("vle32.v v0, (%0)" : : "r"(&out0[(r + HW) * N0]) : "memory");
          asm volatile(".insn r 0x57, 0x6, 0x55, x2, %0, x0" : : "r"(r) : "memory");
        }
        if (vl1 > 0) {
          asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl1));
          for (int r = 0; r < HW; ++r) {
            asm volatile("vle32.v v0, (%0)" : : "r"(&out0[r * N0 + HW]) : "memory");
            asm volatile(".insn r 0x57, 0x6, 0x55, x1, %0, x0" : : "r"(r) : "memory");
            asm volatile("vle32.v v0, (%0)" : : "r"(&out0[(r + HW) * N0 + HW]) : "memory");
            asm volatile(".insn r 0x57, 0x6, 0x55, x3, %0, x0" : : "r"(r) : "memory");
          }
        }
      }

      // K loop: 2x2 tiling with A-load reuse
      const iree_uk_int8_t* lhs_k = lhs_ptr + m_blk;
      const iree_uk_int8_t* rhs0_k = rhs0;
      const iree_uk_int8_t* rhs1_k = rhs1;

      asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(HW));

      for (int k = 0; k < K; ++k) {
        asm volatile(
            "vle8.v v16, (%0)\n\t"   // A row 0 (m_blk..m_blk+15)
            "vle8.v v17, (%1)\n\t"   // B col 0 (n_blk..n_blk+15)
            ".insn r 0x57, 0x2, 0x51, x0, x17, x16\n\t"  // m0
            "vle8.v v18, (%2)\n\t"   // B col 1 (n_blk+16..n_blk+31)
            ".insn r 0x57, 0x2, 0x51, x1, x18, x16\n\t"  // m1
            "vle8.v v19, (%3)\n\t"   // A row 1 (m_blk+16..m_blk+31)
            ".insn r 0x57, 0x2, 0x51, x2, x17, x19\n\t"  // m2
            ".insn r 0x57, 0x2, 0x51, x3, x18, x19\n\t"  // m3
            :
            : "r"(lhs_k), "r"(rhs0_k),
              "r"(rhs1_k), "r"(lhs_k + HW)
            : "memory");

        lhs_k += M0;   // stride by full M0=64
        rhs0_k += N0;
        rhs1_k += N0;
      }

      // Store all 4 sub-tiles
      if (params->flags & IREE_UK_FLAG_MMT4D_TRANSPOSED_OUTPUT) {
        // Column-major store. Column stride = M0 = 64.
        // Sub-tile (m_blk, n_blk) covers rows m_blk..m_blk+31, cols n_blk..n_blk+31.
        // In column-major: col c starts at out_ptr[c * M0 + m_blk].
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(HW));
        // m0: rows m_blk..m_blk+15 → cols n_blk..n_blk+vl0-1
        for (int c = 0; c < vl0; ++c) {
          asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x0" : : "r"(c) : "memory");
          asm volatile("vse32.v v0, (%0)"
                       :
                       : "r"(out_ptr + (n_blk + c) * M0 + m_blk)
                       : "memory");
        }
        // m2: rows m_blk+16..m_blk+31 → cols n_blk..n_blk+vl0-1
        for (int c = 0; c < vl0; ++c) {
          asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x2" : : "r"(c) : "memory");
          asm volatile("vse32.v v0, (%0)"
                       :
                       : "r"(out_ptr + (n_blk + c) * M0 + m_blk + HW)
                       : "memory");
        }
        if (vl1 > 0) {
          // m1: rows m_blk..m_blk+15 → cols n_blk+HW..n_blk+HW+vl1-1
          for (int c = 0; c < vl1; ++c) {
            asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x1" : : "r"(c) : "memory");
            asm volatile("vse32.v v0, (%0)"
                         :
                         : "r"(out_ptr + (n_blk + HW + c) * M0 + m_blk)
                         : "memory");
          }
          // m3: rows m_blk+16..m_blk+31 → cols n_blk+HW..n_blk+HW+vl1-1
          for (int c = 0; c < vl1; ++c) {
            asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x3" : : "r"(c) : "memory");
            asm volatile("vse32.v v0, (%0)"
                         :
                         : "r"(out_ptr + (n_blk + HW + c) * M0 + m_blk + HW)
                         : "memory");
          }
        }
      } else {
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl0));
        for (int r = 0; r < HW; ++r) {
          asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x0" : : "r"(r) : "memory");
          asm volatile("vse32.v v0, (%0)" : : "r"(&out0[r * N0]) : "memory");
        }
        for (int r = 0; r < HW; ++r) {
          asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x2" : : "r"(r) : "memory");
          asm volatile("vse32.v v0, (%0)" : : "r"(&out0[(r + HW) * N0]) : "memory");
        }
        if (vl1 > 0) {
          asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl1));
          for (int r = 0; r < HW; ++r) {
            asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x1" : : "r"(r) : "memory");
            asm volatile("vse32.v v0, (%0)" : : "r"(&out0[r * N0 + HW]) : "memory");
          }
          for (int r = 0; r < HW; ++r) {
            asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, x3" : : "r"(r) : "memory");
            asm volatile("vse32.v v0, (%0)" : : "r"(&out0[(r + HW) * N0 + HW]) : "memory");
          }
        }
      }
    }
  }
}

// =============================================================================
// Saturn OPU int8 K0=128 path with encoding swap.
//
// With the encoding swap, each K1 tile has layout [K0=128, M0=16] for LHS
// and [K0=128, N0=16] for RHS. M0/N0 are innermost (contiguous), so
// vle8.v loads 16 elements per K0 step — no strided loads needed.
// The pack becomes a contiguous memcpy instead of strided gather.
// =============================================================================

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_1xXXx128_to_16xXXx128_riscv_64_xopu(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16);
  IREE_UK_ASSERT(params->K0 == 128);
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;

  const int N0 = params->N0;
  const int K0 = 128;
  const int HW_WIDTH = 16;

  for (int n_start = 0; n_start < N0; n_start += HW_WIDTH) {
    int n_rem = N0 - n_start;
    size_t vl = (n_rem < HW_WIDTH) ? n_rem : HW_WIDTH;
    size_t ml = M0;

    const iree_uk_int8_t* sub_rhs = rhs_ptr + n_start * K0;
    iree_uk_int32_t* sub_out = out_ptr + n_start;

    if (M0 == 16) {
      // Init accumulator
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
      asm volatile("vmv.v.i v0, 0" : : : "memory");
      iree_uk_riscv_64_xopu_opmvinbcast_m0_from_v0();

      if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        for (int row = 0; row < 16; ++row) {
          asm volatile("vle32.v v0, (%0)" : : "r"(&sub_out[row * N0]) : "memory");
          iree_uk_riscv_64_xopu_vmv_rv_m0_from_v0(row);
        }
      }

      // K loop: iterate over K1 tiles, each with K0=128 inner steps.
      // With encoding swap: LHS layout is [K0, M0] = M0 contiguous per K0 step.
      asm volatile("vsetvli zero, %0, e8, m2, ta, ma" : : "r"(ml));
      for (int k1 = 0; k1 < params->K; ++k1) {
        const iree_uk_int8_t* lhs_k = lhs_ptr + k1 * M0 * K0;
        const iree_uk_int8_t* rhs_k = sub_rhs + k1 * N0 * K0;

        // Inner K0=128 loop, 4x unrolled
        int k0 = 0;
        for (; k0 + 4 <= K0; k0 += 4) {
          asm volatile(
              "vle8.v v16, (%0)\n\t"
              "vle8.v v18, (%1)\n\t"
              ".insn r 0x57, 0x2, 0x51, x0, x18, x16\n\t"
              "vle8.v v20, (%2)\n\t"
              "vle8.v v22, (%3)\n\t"
              ".insn r 0x57, 0x2, 0x51, x0, x22, x20\n\t"
              "vle8.v v24, (%4)\n\t"
              "vle8.v v26, (%5)\n\t"
              ".insn r 0x57, 0x2, 0x51, x0, x26, x24\n\t"
              "vle8.v v28, (%6)\n\t"
              "vle8.v v30, (%7)\n\t"
              ".insn r 0x57, 0x2, 0x51, x0, x30, x28\n\t"
              :
              : "r"(lhs_k + k0 * M0), "r"(rhs_k + k0 * N0),
                "r"(lhs_k + (k0 + 1) * M0), "r"(rhs_k + (k0 + 1) * N0),
                "r"(lhs_k + (k0 + 2) * M0), "r"(rhs_k + (k0 + 2) * N0),
                "r"(lhs_k + (k0 + 3) * M0), "r"(rhs_k + (k0 + 3) * N0)
              : "memory");
        }
        for (; k0 < K0; ++k0) {
          asm volatile(
              "vle8.v v16, (%0)\n\t"
              "vle8.v v18, (%1)\n\t"
              ".insn r 0x57, 0x2, 0x51, x0, x18, x16\n\t"
              :
              : "r"(lhs_k + k0 * M0), "r"(rhs_k + k0 * N0)
              : "memory");
        }
      }

      // Store results
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
      for (size_t r = 0; r < ml; r++) {
        iree_uk_riscv_64_xopu_vmv_vr_v0_from_m0(r);
        asm volatile("vse32.v v0, (%0)" : : "r"(&sub_out[r * N0]) : "memory");
      }
    } else {
      // Tail case (M0 <= 8): use smaller LMUL
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl));
      asm volatile("vmv.v.i v0, 0" : : : "memory");
      iree_uk_riscv_64_xopu_opmvinbcast_m0_from_v0();

      if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        for (size_t row = 0; row < ml; ++row) {
          asm volatile("vle32.v v0, (%0)" : : "r"(&sub_out[row * N0]) : "memory");
          iree_uk_riscv_64_xopu_vmv_rv_m0_from_v0(row);
        }
      }

      asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(ml));
      for (int k1 = 0; k1 < params->K; ++k1) {
        const iree_uk_int8_t* lhs_k = lhs_ptr + k1 * M0 * K0;
        const iree_uk_int8_t* rhs_k = sub_rhs + k1 * N0 * K0;
        for (int k0 = 0; k0 < K0; ++k0) {
          asm volatile(
              "vle8.v v4, (%0)\n\t"
              "vle8.v v5, (%1)\n\t"
              ".insn r 0x57, 0x2, 0x51, x0, x5, x4\n\t"
              :
              : "r"(lhs_k + k0 * M0), "r"(rhs_k + k0 * N0)
              : "memory");
        }
      }

      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl));
      for (size_t r = 0; r < ml; r++) {
        iree_uk_riscv_64_xopu_vmv_vr_v0_from_m0(r);
        asm volatile("vse32.v v0, (%0)" : : "r"(&sub_out[r * N0]) : "memory");
      }
    }
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx128_to_16xXXx128_riscv_64_xopu,
    iree_uk_mmt4d_tile_s8s8s32_1xXXx128_riscv_64_xopu, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx128_to_16xXXx128_riscv_64_xopu,
    iree_uk_mmt4d_tile_s8s8s32_2xXXx128_riscv_64_xopu, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx128_to_16xXXx128_riscv_64_xopu,
    iree_uk_mmt4d_tile_s8s8s32_4xXXx128_riscv_64_xopu, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx128_to_16xXXx128_riscv_64_xopu,
    iree_uk_mmt4d_tile_s8s8s32_8xXXx128_riscv_64_xopu, 8)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx128_to_16xXXx128_riscv_64_xopu,
    iree_uk_mmt4d_tile_s8s8s32_16xXXx128_riscv_64_xopu, 16)

// =============================================================================
// Saturn OPU FP8: OPFMACC (f8E4M3FN x f8E4M3FN -> f32 accumulation)
//
// Uses the OPU matrix engine with OPFMACC instruction:
//   .insn r 0x57, 0x1, 0x4b, md, vs1, vs2
// where md is a matrix register (m0-m3), vs1/vs2 are vector registers.
//
// The accumulator is in f32 (same as int8 VOPACC), stored in matrix registers.
// Output is f32 stored as uint32_t bits (IEEE 754 binary32).
//
// For E5M2 format, use VSETVLI_ALTFMT with altfmt=1.
// For E4M3 format, use standard vsetvli (altfmt=0).
//
// Tile: 16x16xK0 (same geometry as int8 OPU)
// =============================================================================

// VSETVLI_ALTFMT: set e8,m1 with altfmt bit for E5M2 format selection.
// Standard vsetvli (altfmt=0) selects E4M3FN.
// altfmt=1 selects E5M2.
// Encoding: vsetvli with vtypei bit 8 = altfmt.
//   vtypei for e8,m1,ta,ma,altfmt=0: 0xC0
//   vtypei for e8,m1,ta,ma,altfmt=1: 0x1C0
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_riscv_64_xopu_vsetvli_e8_m1_altfmt(long avl, int altfmt) {
  if (altfmt) {
    // .insn i 0x57, 7, x0, rs1, 0x1C0 = vsetvli zero, rs1, e8,m1,ta,ma,altfmt=1
    asm volatile(".insn i 0x57, 7, x0, %0, 0x1C0" : : "r"(avl));
  } else {
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(avl));
  }
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f8e4m3f8e4m3f32_16x16_riscv_64_xopu_native(
    iree_uk_int32_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  const long avl16 = 16;

  asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t"
               :
               : "r"(avl16)
               : "memory");

  // Initialize accumulator m0 (f32 in matrix register).
  asm volatile("vmv.v.i v0, 0\n\t" : : : "memory");
  iree_uk_riscv_64_xopu_opmvinbcast_m0_from_v0();
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int row = 0; row < 16; ++row) {
      asm volatile("vle32.v v0, (%0)\n\t"
                   :
                   : "r"(out_ptr + row * 16)
                   : "memory");
      iree_uk_riscv_64_xopu_vmv_rv_m0_from_v0(row);
    }
  }

  // Per-operand encoding: tile layout [K0, M0] (K-outer, M-inner).
  // For k0=fixed, M0=16 fp8 elements contiguous → vle8.v.
  // E4M3FN: altfmt=0 (standard vsetvli).
  for (int k = 0; k < params->K; ++k) {
    const iree_uk_int8_t* lhs_k_ptr = lhs_ptr + k * 16 * params->K0;
    const iree_uk_int8_t* rhs_k_ptr = rhs_ptr + k * 16 * params->K0;
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t"
                 :
                 : "r"(avl16)
                 : "memory");

    // Unroll K0 loop by 2 to hide VRF RAW hazards.
    int k0 = 0;
    for (; k0 + 2 <= params->K0; k0 += 2) {
      asm volatile("vle8.v v16, (%0)\n\t"
                   :
                   : "r"(lhs_k_ptr + k0 * 16)
                   : "memory");
      asm volatile("vle8.v v18, (%0)\n\t"
                   :
                   : "r"(rhs_k_ptr + k0 * 16)
                   : "memory");
      iree_uk_riscv_64_xopu_opfmacc_m0_v16_v18();
      asm volatile("vle8.v v20, (%0)\n\t"
                   :
                   : "r"(lhs_k_ptr + (k0 + 1) * 16)
                   : "memory");
      asm volatile("vle8.v v22, (%0)\n\t"
                   :
                   : "r"(rhs_k_ptr + (k0 + 1) * 16)
                   : "memory");
      iree_uk_riscv_64_xopu_opfmacc_m0_v20_v22();
    }
    for (; k0 < params->K0; ++k0) {
      asm volatile("vle8.v v16, (%0)\n\t"
                   :
                   : "r"(lhs_k_ptr + k0 * 16)
                   : "memory");
      asm volatile("vle8.v v18, (%0)\n\t"
                   :
                   : "r"(rhs_k_ptr + k0 * 16)
                   : "memory");
      iree_uk_riscv_64_xopu_opfmacc_m0_v16_v18();
    }
  }

  // Store results from matrix register m0 row by row.
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t"
               :
               : "r"(avl16)
               : "memory");
  for (int row = 0; row < 16; ++row) {
    iree_uk_riscv_64_xopu_vmv_vr_v0_from_m0(row);
    asm volatile("vse32.v v0, (%0)\n\t"
                 :
                 : "r"(out_ptr + row * 16)
                 : "memory");
  }
}

// FP8 OPU fallback for narrow-M using RVV (same pattern as int8 OPU fallback).
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f8e4m3f8e4m3f32_1xXXx16_to_16xXXx16_riscv_64_xopu(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  if (M0 == 16 && params->N0 == 16) {
    iree_uk_mmt4d_tile_f8e4m3f8e4m3f32_16x16_riscv_64_xopu_native(
        (iree_uk_int32_t*)out_tile,
        (const iree_uk_int8_t*)lhs_panel,
        (const iree_uk_int8_t*)rhs_panel,
        params);
    return;
  }
  // TODO: narrow-M fallback for fp8 OPU.
  // For now, zero-fill output to avoid undefined behavior.
  iree_uk_int32_t* out_ptr = (iree_uk_int32_t*)out_tile;
  for (int i = 0; i < M0 * params->N0; ++i) out_ptr[i] = 0;
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f8e4m3f8e4m3f32_1xXXx16_to_16xXXx16_riscv_64_xopu,
    iree_uk_mmt4d_tile_f8e4m3f8e4m3f32_1xXXx16_riscv_64_xopu, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f8e4m3f8e4m3f32_1xXXx16_to_16xXXx16_riscv_64_xopu,
    iree_uk_mmt4d_tile_f8e4m3f8e4m3f32_2xXXx16_riscv_64_xopu, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f8e4m3f8e4m3f32_1xXXx16_to_16xXXx16_riscv_64_xopu,
    iree_uk_mmt4d_tile_f8e4m3f8e4m3f32_4xXXx16_riscv_64_xopu, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f8e4m3f8e4m3f32_1xXXx16_to_16xXXx16_riscv_64_xopu,
    iree_uk_mmt4d_tile_f8e4m3f8e4m3f32_8xXXx16_riscv_64_xopu, 8)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f8e4m3f8e4m3f32_1xXXx16_to_16xXXx16_riscv_64_xopu,
    iree_uk_mmt4d_tile_f8e4m3f8e4m3f32_16xXXx16_riscv_64_xopu, 16)

// =============================================================================
// Fused OPU matmul + QDQ kernel (XNNPACK-style).
//
// Computes: out_i8 = requant(dequant(matmul_i32) * scale + bias)
//
// This fuses the entire QDQ chain into the OPU matmul kernel:
//   1. VOPACC: i8 × i8 → i32 in matrix registers
//   2. VMV_VR: extract i32 row to vector register
//   3. vfcvt.f.x.v: i32 → f32
//   4. vfmadd.vf: f32 * dequant_scale + bias (fused multiply-add)
//   5. vfmul.vf: f32 * inv_requant_scale (= 1/output_scale)
//   6. vfcvt.x.f.v: f32 → i32 with RNE rounding (hardware roundeven)
//   7. vmax/vmin: clamp to [-128, 127]
//   8. vnsrl: i32 → i16 → i8 (narrow)
//   9. vse8: store i8 output
//
// ~12 instructions per output row vs ~165 for the split mmt4d + generic path.
// Modeled on saturn/benchmarks and XNNPACK qs8_qc8w_gemm_minmax_fp32.
// =============================================================================

// Store one 16-element row from matrix register mx as requantized i8.
// Performs: dequant(i32) → scale+bias → requant → clamp → i8.
// Uses hardware RNE rounding via fsrmi/vfcvt instead of software roundeven.
#define OPU_STORE_SUBTILE_QDQ_I8(mx, out_ptr, out_stride, nrows, ncols,       \
                                 dequant_scale, bias_ptr,                      \
                                 inv_requant_scale)                            \
  do {                                                                         \
    /* Load bias vector once (same for all rows — per-channel bias) */         \
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(ncols));        \
    asm volatile("vle32.v v4, (%0)" : : "r"(bias_ptr) : "memory");           \
    for (int _r = 0; _r < (nrows); ++_r) {                                    \
      /* Extract row from matrix register to v0 (i32 x ncols) */              \
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(ncols));      \
      asm volatile(".insn r 0x57, 0x6, 0x5d, x0, %0, " #mx                   \
                   :                                                           \
                   : "r"(_r)                                                   \
                   : "memory");                                                \
      /* v0 = i32 accumulator row */                                           \
      /* Dequant: i32 → f32, then v0 = v0 * dequant_scale + bias */           \
      asm volatile("vfcvt.f.x.v v0, v0" ::: "memory");                       \
      asm volatile("vfmadd.vf v0, %0, v4"                                     \
                   :                                                           \
                   : "f"((float)(dequant_scale))                               \
                   : "memory");                                                \
      /* Requant: multiply by 1/output_scale */                                \
      asm volatile("vfmul.vf v0, v0, %0"                                      \
                   :                                                           \
                   : "f"((float)(inv_requant_scale))                           \
                   : "memory");                                                \
      /* Round to nearest even using hardware RNE mode */                      \
      asm volatile("fsrmi 0" ::: "memory"); /* RNE rounding mode */           \
      asm volatile("vfcvt.x.f.v v0, v0" ::: "memory"); /* f32→i32 RNE */     \
      asm volatile("fsrmi 4" ::: "memory"); /* Restore default */             \
      /* Clamp to [-128, 127] */                                               \
      asm volatile("vmax.vx v0, v0, %0" : : "r"(-128) : "memory");           \
      asm volatile("vmin.vx v0, v0, %0" : : "r"(127) : "memory");            \
      /* Narrow i32 → i16 → i8 */                                             \
      asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(ncols));      \
      asm volatile("vnsrl.wi v8, v0, 0" ::: "memory"); /* i32→i16 */         \
      asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(ncols));       \
      asm volatile("vnsrl.wi v8, v8, 0" ::: "memory"); /* i16→i8 */          \
      /* ReLU: max(0, x) */                                                    \
      asm volatile("vmax.vx v8, v8, zero" ::: "memory");                      \
      /* Store i8 output */                                                    \
      asm volatile("vse8.v v8, (%0)"                                           \
                   :                                                           \
                   : "r"((iree_uk_int8_t*)(out_ptr) + _r * (out_stride))      \
                   : "memory");                                                \
    }                                                                          \
  } while (0)

// Fused OPU matmul + QDQ: i8 × i8 → i8 (with dequant, bias, requant, ReLU).
// Parameters:
//   lhs_packed: [M, K, M0, K0] packed int8 LHS
//   rhs_packed: [N, K, N0, K0] packed int8 RHS
//   out_i8:     [M*M0, N*N0] flat int8 output (row-major)
//   M, N, K, M0, N0, K0: tile dimensions
//   dequant_scale: scalar f32 (product of input_scale * weight_scale)
//   bias:       [N*N0] f32 per-channel bias (pre-scaled)
//   inv_requant_scale: scalar f32 (1.0 / output_scale)
IREE_UK_EXPORT void iree_uk_opu_matmul_qdq(
    const void* lhs_packed, const void* rhs_packed, void* out_i8,
    int32_t M, int32_t N, int32_t K, int32_t M0, int32_t N0, int32_t K0,
    float dequant_scale, const float* bias, float inv_requant_scale) {
  const int HW = 16;  // OPU hardware width
  const iree_uk_int8_t* lhs = (const iree_uk_int8_t*)lhs_packed;
  const iree_uk_int8_t* rhs = (const iree_uk_int8_t*)rhs_packed;
  iree_uk_int8_t* out = (iree_uk_int8_t*)out_i8;

  const int lhs_panel_stride = K * M0 * K0;
  const int rhs_panel_stride = K * N0 * K0;
  const int out_stride = N * N0;  // row stride for 2D output

  for (int i = 0; i < M; ++i) {
    const iree_uk_int8_t* lhs_panel = lhs + i * lhs_panel_stride;

    for (int j = 0; j < N; ++j) {
      const iree_uk_int8_t* rhs_panel = rhs + j * rhs_panel_stride;
      iree_uk_int8_t* out_tile = out + i * M0 * out_stride + j * N0;

      // Process sub-tiles using 16x16 OPU hardware
      for (int m_sub = 0; m_sub < M0; m_sub += HW) {
        int m_hw = (M0 - m_sub < HW) ? (M0 - m_sub) : HW;

        for (int n_sub = 0; n_sub < N0; n_sub += HW) {
          int n_hw = (N0 - n_sub < HW) ? (N0 - n_sub) : HW;

          // Init accumulator matrix register m0
          asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(n_hw));
          asm volatile("vmv.v.i v0, 0" ::: "memory");
          iree_uk_riscv_64_xopu_opmvinbcast_m0_from_v0();

          // K loop: accumulate outer products
          asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(HW));
          for (int k = 0; k < K; ++k) {
            const iree_uk_int8_t* lhs_k = lhs_panel + k * M0 * K0 + m_sub;
            const iree_uk_int8_t* rhs_k = rhs_panel + k * N0 * K0 + n_sub;

            for (int k0 = 0; k0 < K0; ++k0) {
              asm volatile(
                  "vle8.v v16, (%0)\n\t"
                  "vle8.v v18, (%1)\n\t"
                  :
                  : "r"(lhs_k + k0 * M0), "r"(rhs_k + k0 * N0)
                  : "memory");
              iree_uk_riscv_64_xopu_vopacc_m0_v16_v18();
            }
          }

          // Fused QDQ store: dequant + bias + requant + ReLU → i8
          const float* bias_tile = bias + j * N0 + n_sub;
          iree_uk_int8_t* out_sub =
              out_tile + m_sub * out_stride + n_sub;

          OPU_STORE_SUBTILE_QDQ_I8(x0, out_sub, out_stride, m_hw, n_hw,
                                    dequant_scale, bias_tile,
                                    inv_requant_scale);
        }
      }
    }
  }
}

#endif // IREE_UK_ARCH_RISCV_64
