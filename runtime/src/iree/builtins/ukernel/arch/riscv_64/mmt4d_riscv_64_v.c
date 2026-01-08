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


// --- IME Instruction Macros ---
// vmadot: Opcode 0x2b, Funct3 1 (SS), Funct7 0x38
#define VMADOT_I8(vd, vs1, vs2) \
    asm volatile(".insn r 0x2b, 3, 0x71, " vd ", " vs1 ", " vs2)

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_4x4x8_riscv_64_ime_impl(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  
  IREE_UK_ASSERT(M0 == 4 || M0 == 8);
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;


  // =======================================================================
  // CASE 1: M0 = 4 (Single Tile)
  // =======================================================================
  if (M0 == 4) {
    
    // 1. Initialize Accumulator (v8)
    // Destination v8 is Even. OK.
    size_t vl_acc = 16;
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl_acc));

    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        asm volatile("vle32.v v8, (%0)" : : "r"(out_ptr));
    } else {
        asm volatile("vmv.v.i v8, 0");
    }

    // 2. Main Loop
    // Hardware requires SEW=8, VL=32 for the 4x4x8 unit
    size_t vl_in = 32;
    
    
    for (int k = 0; k < params->K; k++) {
      asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(vl_in));

      // Load LHS (M0xK packed -> 4x8 = 32 bytes)
      asm volatile("vle8.v v0, (%0)" : : "r"(lhs_ptr));
      
      // Load RHS (N0xK packed -> 4x8 = 32 bytes)
      asm volatile("vle8.v v4, (%0)" : : "r"(rhs_ptr));

      VMADOT_I8("v8", "v0", "v4");
      
      // LHS Tile: 4 * 8 * 1 byte = 32 bytes
      // RHS Tile: 4 * 8 * 1 byte = 32 bytes
      lhs_ptr += 32;
      rhs_ptr += 32;
    }

    // 3. Store Result
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl_acc));
    asm volatile("vse32.v v8, (%0)" : : "r"(out_ptr) : "memory");
  }
  // =======================================================================
  // CASE 2: M0 = 8 (Double Tile)
  // =======================================================================
  else if (M0 == 8) {
    
    // 1. Init Accumulators (v8, v10)
    size_t vl_acc = 16;
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl_acc));

    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        asm volatile("vle32.v v8,  (%0)" : : "r"(out_ptr));
        asm volatile("vle32.v v10, (%0)" : : "r"(&out_ptr[16]));
    } else {
        asm volatile("vmv.v.i v8, 0");
        asm volatile("vmv.v.i v10, 0");
    }

    // 2. Loop
    size_t vl_in = 32;
    for (int k = 0; k < params->K; k++) { // Loop 16 times
      asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(vl_in));
      
      // 1. Load LHS Block (64 bytes total)
      // Top half (Rows 0-3): 32 bytes -> v0
      asm volatile("vle8.v v0, (%0)" : : "r"(lhs_ptr));
      // Bottom half (Rows 4-7): 32 bytes -> v2
      asm volatile("vle8.v v2, (%0)" : : "r"(lhs_ptr + 32));
      
      // 2. Load RHS Block (32 bytes total)
      // Rows 0-7: 32 bytes -> v4
      asm volatile("vle8.v v4, (%0)" : : "r"(rhs_ptr));
      
      // 3. Compute
      VMADOT_I8("v8", "v0", "v4");   // Top
      VMADOT_I8("v10", "v2", "v4");  // Bottom
      
      // 4. Advance Pointers by Block Size
      lhs_ptr += 64; 
      rhs_ptr += 32;
  }

    // 3. Store
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl_acc));
    asm volatile("vse32.v v8,  (%0)" : : "r"(out_ptr));
    asm volatile("vse32.v v10, (%0)" : : "r"(&out_ptr[16]));
  }
}


// --- IME Registrations (Spacemit K1) ---

// Registration wrapper function
void iree_uk_mmt4d_tile_s8s8s32_4x4x8_riscv_64_ime(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
    
    int M0 = params->M0 > 0 ? params->M0 : 4;
    iree_uk_mmt4d_tile_s8s8s32_4x4x8_riscv_64_ime_impl(out_tile, lhs_panel, rhs_panel, params, M0);
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