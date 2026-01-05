// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <riscv_vector.h>

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"
#include "bme.h"

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f32f32f32_1xXXx1_to_7xXXx1_riscv_64_v(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 7);
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;

  vfloat32m4_t acc0, acc1, acc2, acc3, acc4, acc5, acc6;

  int N0 = params->N0;
  size_t vl = N0;
  if (M0 == 1) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
    } else {
      acc0 = __riscv_vfmv_v_f_f32m4(0.0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat32m4_t rhs = __riscv_vle32_v_f32m4(rhs_ptr, vl);
      rhs_ptr += N0;
      float lhs = *lhs_ptr++;
      acc0 = __riscv_vfmacc_vf_f32m4(acc0, lhs, rhs, vl);
    }
    __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
  } else if (M0 == 2) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_f32m4(out_ptr + N0, vl);
    } else {
      acc0 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc1 = __riscv_vfmv_v_f_f32m4(0.0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat32m4_t rhs = __riscv_vle32_v_f32m4(rhs_ptr, vl);
      rhs_ptr += N0;
      float lhs[2];
      IREE_UK_UNROLL for (int i = 0; i < M0; ++i) { lhs[i] = *lhs_ptr++; }
      acc0 = __riscv_vfmacc_vf_f32m4(acc0, lhs[0], rhs, vl);
      acc1 = __riscv_vfmacc_vf_f32m4(acc1, lhs[1], rhs, vl);
    }
    __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0, acc1, vl);
  } else if (M0 == 4) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_f32m4(out_ptr + N0, vl);
      acc2 = __riscv_vle32_v_f32m4(out_ptr + N0 * 2, vl);
      acc3 = __riscv_vle32_v_f32m4(out_ptr + N0 * 3, vl);
    } else {
      acc0 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc1 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc2 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc3 = __riscv_vfmv_v_f_f32m4(0.0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat32m4_t rhs = __riscv_vle32_v_f32m4(rhs_ptr, vl);
      rhs_ptr += N0;
      float lhs[4];
      IREE_UK_UNROLL for (int i = 0; i < M0; ++i) { lhs[i] = *lhs_ptr++; }
      acc0 = __riscv_vfmacc_vf_f32m4(acc0, lhs[0], rhs, vl);
      acc1 = __riscv_vfmacc_vf_f32m4(acc1, lhs[1], rhs, vl);
      acc2 = __riscv_vfmacc_vf_f32m4(acc2, lhs[2], rhs, vl);
      acc3 = __riscv_vfmacc_vf_f32m4(acc3, lhs[3], rhs, vl);
    }
    __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0, acc1, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 2, acc2, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 3, acc3, vl);
  } else if (M0 == 7) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_f32m4(out_ptr + N0, vl);
      acc2 = __riscv_vle32_v_f32m4(out_ptr + N0 * 2, vl);
      acc3 = __riscv_vle32_v_f32m4(out_ptr + N0 * 3, vl);
      acc4 = __riscv_vle32_v_f32m4(out_ptr + N0 * 4, vl);
      acc5 = __riscv_vle32_v_f32m4(out_ptr + N0 * 5, vl);
      acc6 = __riscv_vle32_v_f32m4(out_ptr + N0 * 6, vl);
    } else {
      acc0 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc1 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc2 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc3 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc4 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc5 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc6 = __riscv_vfmv_v_f_f32m4(0.0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat32m4_t rhs = __riscv_vle32_v_f32m4(rhs_ptr, vl);
      rhs_ptr += N0;
      float lhs[7];
      IREE_UK_UNROLL for (int i = 0; i < M0; ++i) { lhs[i] = *lhs_ptr++; }
      acc0 = __riscv_vfmacc_vf_f32m4(acc0, lhs[0], rhs, vl);
      acc1 = __riscv_vfmacc_vf_f32m4(acc1, lhs[1], rhs, vl);
      acc2 = __riscv_vfmacc_vf_f32m4(acc2, lhs[2], rhs, vl);
      acc3 = __riscv_vfmacc_vf_f32m4(acc3, lhs[3], rhs, vl);
      acc4 = __riscv_vfmacc_vf_f32m4(acc4, lhs[4], rhs, vl);
      acc5 = __riscv_vfmacc_vf_f32m4(acc5, lhs[5], rhs, vl);
      acc6 = __riscv_vfmacc_vf_f32m4(acc6, lhs[6], rhs, vl);
    }
    __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0, acc1, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 2, acc2, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 3, acc3, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 4, acc4, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 5, acc5, vl);
    __riscv_vse32_v_f32m4(out_ptr + N0 * 6, acc6, vl);
  }
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_16xXXx1_riscv_64_v(
    void* IREE_UK_RESTRICT out_tile,
    const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 16);
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;

  const int N0 = params->N0;
  const int K = params->K;
  size_t ml = M0;
  size_t vl = N0;

  // Performance case for M0=16 (LMUL=8)
  if (M0 == 16) {
    // init m0 to zero (LMUL=8)
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vmv.v.i v0, 0");
    OPMVINBCAST(m0, v0);

    // K-loop unrolled by 2
    size_t k = 0;
    while (k + 2 <= K) {
      asm volatile("vsetvli zero, %0, e8, m2, ta, ma" : : "r"(ml));  // ml=16
      asm volatile("vle8.v v16, (%0)" : : "r"(&lhs_ptr[k * M0]));
      asm volatile("vsetvli zero, %0, e8, m2, ta, ma" : : "r"(vl));  // vl=N0
      asm volatile("vle8.v v18, (%0)" : : "r"(&rhs_ptr[k * N0]));
      VOPACC(m0, v18, v16);
      k++;
      asm volatile("vsetvli zero, %0, e8, m2, ta, ma" : : "r"(ml));  // ml=16
      asm volatile("vle8.v v20, (%0)" : : "r"(&lhs_ptr[k * M0]));
      asm volatile("vsetvli zero, %0, e8, m2, ta, ma" : : "r"(vl));  // vl=N0
      asm volatile("vle8.v v22, (%0)" : : "r"(&rhs_ptr[k * N0]));
      VOPACC(m0, v22, v20);
      k++;
    }
    if (k < K) {
      asm volatile("vsetvli zero, %0, e8, m2, ta, ma" : : "r"(ml));  // ml=16
      asm volatile("vle8.v v16, (%0)" : : "r"(&lhs_ptr[k * M0]));
      asm volatile("vsetvli zero, %0, e8, m2, ta, ma" : : "r"(vl));  // vl=N0
      asm volatile("vle8.v v18, (%0)" : : "r"(&rhs_ptr[k * N0]));
      VOPACC(m0, v18, v16);
    }

    // store results
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    for (size_t r = 0; r < ml; r++) {  // ml=16
      VMV_VR(v0, r, m0);
      asm volatile("vse32.v v0, (%0)" : : "r"(&out_ptr[r * N0]));
    }
  }
  // Tail case for M0 < 16 (using LMUL=4)
  else {
    // 1. Initialize accumulators to ZERO (LMUL=4)
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl));
    asm volatile("vmv.v.i v0, 0");
    OPMVINBCAST(m3, v0);  // Initialize m3 to zero

    // 2. Main K-loop
    for (int k = 0; k < K; ++k) {
      asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(ml));
      asm volatile("vle8.v v5, (%0)" : : "r"(&lhs_ptr[k * M0]));

      asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(vl));
      asm volatile("vle8.v v4, (%0)" : : "r"(&rhs_ptr[k * N0]));

      VOPACC(m3, v4, v5);
    }

    // 3. Store results
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl));
    for (size_t r = 0; r < ml; r++) {
      VMV_VR(v0, r, m3);
      asm volatile("vse32.v v0, (%0)" : : "r"(&out_ptr[r * N0]));
    }
  }
}


IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_4x4x8_riscv_64_ime_impl(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  
  IREE_UK_ASSERT(M0 == 4 || M0 == 8);
  const int8_t* lhs_ptr = (const int8_t*)lhs_panel;
  const int8_t* rhs_ptr = (const int8_t*)rhs_panel;
  int32_t* out_ptr = (int32_t*)out_tile;
  int K = params->K;

  // =======================================================================
  // CASE 1: M0 = 4 (Single Tile)
  // =======================================================================
  if (M0 == 4) {
    // Registers:
    // v8-v9: Accumulator (4x4 int32) - Constraint: Even register
    // v0:    LHS (4x8 int8)
    // v4:    RHS (8x4 int8)

    // 1. Initialize Accumulator
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        // Load 64 bytes (4x4 int32)
        // m2 = 2 registers = 512 bits = 64 bytes
        __asm__ volatile(
            "vsetvli zero, zero, e32, m2, ta, ma \n"
            "vle32.v v8, (%0)                    \n"
            : : "r"(out_ptr) : "v8", "v9"
        );
    } else {
        __asm__ volatile(
            "vsetvli zero, zero, e32, m2, ta, ma \n"
            "vmv.v.i v8, 0                       \n"
            : : : "v8", "v9"
        );
    }

    // 2. Main Loop (Unrolled x2 for pipeline efficiency if K is large enough)
    // For simplicity and code size, we use a simple loop here. 
    // The hardware processes 8 K-elements per instruction.
    for (; K >= 8; K -= 8) {
        __asm__ volatile(
            // Config: e8, m1 (32 bytes per register)
            "vsetvli zero, zero, e8, m1, ta, ma  \n"
            
            "vle8.v  v0, (%[lhs])                \n" // Load A (32B)
            "vle8.v  v4, (%[rhs])                \n" // Load B (32B)
            
            // vmadot v8, v0, v4
            // Instruction encoding: 111000 (func7) | 001 (SS) | ...
            // Using assembler mnemonic if available, else .insn
            "vmadot  v8, v0, v4                  \n"
            
            : 
            : [lhs] "r"(lhs_ptr), [rhs] "r"(rhs_ptr)
            : "v0", "v4", "v8", "v9"
        );
        lhs_ptr += 32;
        rhs_ptr += 32;
    }

    // 3. Store Result
    __asm__ volatile(
        "vsetvli zero, zero, e32, m2, ta, ma \n"
        "vse32.v v8, (%0)                    \n"
        : : "r"(out_ptr) : "memory"
    );
  } 
  // =======================================================================
  // CASE 2: M0 = 8 (Two Tiles Stacked Vertically)
  // =======================================================================
  else if (M0 == 8) {
    // We compute two 4x4 blocks: Top (rows 0-3) and Bottom (rows 4-7).
    // They share the same RHS (B) data.
    
    // Registers:
    // v8-v9:   Accumulator Top
    // v10-v11: Accumulator Bottom
    // v0:      LHS Top
    // v1:      LHS Bottom (Separate register because v0 is m1)
    // v4:      RHS (Shared)

    // 1. Initialize Accumulators
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        __asm__ volatile(
            "vsetvli zero, zero, e32, m2, ta, ma \n"
            "vle32.v v8,  (%0)                   \n" // Top 64B
            "addi    t0, %0, 64                  \n"
            "vle32.v v10, (t0)                   \n" // Bot 64B
            : : "r"(out_ptr) : "v8", "v9", "v10", "v11", "t0"
        );
    } else {
        __asm__ volatile(
            "vsetvli zero, zero, e32, m2, ta, ma \n"
            "vmv.v.i v8, 0                       \n"
            "vmv.v.i v10, 0                      \n"
            : : : "v8", "v9", "v10", "v11"
        );
    }

    // 2. Main Loop
    for (; K >= 8; K -= 8) {
        __asm__ volatile(
            "vsetvli zero, zero, e8, m1, ta, ma  \n"
            
            // Load LHS (Top and Bottom)
            // LHS is packed as 8x8 (M0xK0), so it is contiguous 64 bytes.
            "vle8.v  v0, (%[lhs])                \n"
            "addi    t0, %[lhs], 32              \n"
            "vle8.v  v1, (t0)                    \n"
            
            // Load RHS (Shared)
            "vle8.v  v4, (%[rhs])                \n"
            
            // Compute Top Tile
            "vmadot  v8, v0, v4                  \n"
            
            // Compute Bottom Tile
            "vmadot  v10, v1, v4                 \n"
            
            : 
            : [lhs] "r"(lhs_ptr), [rhs] "r"(rhs_ptr)
            : "v0", "v1", "v4", "v8", "v9", "v10", "v11", "t0"
        );
        lhs_ptr += 64; // 8x8 bytes
        rhs_ptr += 32; // 4x8 bytes (RHS N0=4)
    }

    // 3. Store Results
    __asm__ volatile(
        "vsetvli zero, zero, e32, m2, ta, ma \n"
        "vse32.v v8,  (%0)                   \n"
        "addi    t0, %0, 64                  \n"
        "vse32.v v10, (t0)                   \n"
        : : "r"(out_ptr) : "memory", "t0"
    );
  }
}


IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1xXXx1_to_7xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_f32f32f32_1xXXx1_riscv_64_v, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1xXXx1_to_7xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_f32f32f32_2xXXx1_riscv_64_v, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1xXXx1_to_7xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_f32f32f32_4xXXx1_riscv_64_v, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1xXXx1_to_7xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_f32f32f32_7xXXx1_riscv_64_v, 7)

// OPU Saturn
// Point all s8s8s32 tiles to the new generic function
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_16xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_riscv_64_v, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_16xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_s8s8s32_2xXXx1_riscv_64_v, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_16xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_s8s8s32_4xXXx1_riscv_64_v, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_16xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_s8s8s32_8xXXx1_riscv_64_v, 8)
// Add the new M0=16 tile
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_16xXXx1_riscv_64_v,
    iree_uk_mmt4d_tile_s8s8s32_16xXXx1_riscv_64_v, 16)

// --- IME Registrations (Spacemit K1) ---
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
  iree_uk_mmt4d_tile_s8s8s32_4x4x8_riscv_64_ime_impl,   
  iree_uk_mmt4d_tile_s8s8s32_4xXXx8_riscv_64_ime, 4)    

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
  iree_uk_mmt4d_tile_s8s8s32_4x4x8_riscv_64_ime_impl,   
  iree_uk_mmt4d_tile_s8s8s32_8xXXx8_riscv_64_ime, 8)    