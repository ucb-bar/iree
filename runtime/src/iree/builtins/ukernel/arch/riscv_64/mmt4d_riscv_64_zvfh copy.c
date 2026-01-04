// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <riscv_vector.h>

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"

// Shared implementation for f16f16f16 and f16f16f32.
// In the f16f16f16 case, intermediate roundings are skipped. This function
// should only be used if IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS is set.
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f16f16fXX_1xXXx1_to_7xXXx1_riscv_64_zvfh(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, iree_uk_type_t acc_type, int M0) {
  
  // Relax assertion to allow M0=8
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8);
  const _Float16* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const _Float16* IREE_UK_RESTRICT rhs_ptr = rhs_panel;

  vfloat32m4_t acc0, acc1, acc2, acc3, acc4, acc5, acc6;
  int N0 = params->N0;
  size_t vl = N0;

  if (M0 == 1) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      if (acc_type == IREE_UK_TYPE_FLOAT_32) {
        float* IREE_UK_RESTRICT out_ptr = out_tile;
        acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
      } else {
        _Float16* IREE_UK_RESTRICT out_ptr = out_tile;
        acc0 =
            __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(out_ptr, vl), vl);
      }
    } else {
      acc0 = __riscv_vfmv_v_f_f32m4(0.0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat16m2_t rhs = __riscv_vle16_v_f16m2(rhs_ptr, vl);
      rhs_ptr += N0;
      _Float16 lhs0 = *lhs_ptr++;

      acc0 = __riscv_vfwmacc_vf_f32m4(acc0, lhs0, rhs, vl);
    }
    if (acc_type == IREE_UK_TYPE_FLOAT_32) {
      float* IREE_UK_RESTRICT out_ptr = out_tile;
      __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
    } else {
      _Float16* IREE_UK_RESTRICT out_ptr = out_tile;
      __riscv_vse16_v_f16m2(out_ptr, __riscv_vfncvt_f_f_w_f16m2(acc0, vl), vl);
    }
  } else if (M0 == 2) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      if (acc_type == IREE_UK_TYPE_FLOAT_32) {
        float* IREE_UK_RESTRICT out_ptr = out_tile;
        acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
        acc1 = __riscv_vle32_v_f32m4(out_ptr + N0, vl);
      } else {
        _Float16* IREE_UK_RESTRICT out_ptr = out_tile;
        acc0 =
            __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(out_ptr, vl), vl);
        acc1 = __riscv_vfwcvt_f_f_v_f32m4(
            __riscv_vle16_v_f16m2(out_ptr + N0, vl), vl);
      }
    } else {
      acc0 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc1 = __riscv_vfmv_v_f_f32m4(0.0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat16m2_t rhs = __riscv_vle16_v_f16m2(rhs_ptr, vl);
      rhs_ptr += N0;
      _Float16 lhs0 = *lhs_ptr++;
      _Float16 lhs1 = *lhs_ptr++;

      acc0 = __riscv_vfwmacc_vf_f32m4(acc0, lhs0, rhs, vl);
      acc1 = __riscv_vfwmacc_vf_f32m4(acc1, lhs1, rhs, vl);
    }
    if (acc_type == IREE_UK_TYPE_FLOAT_32) {
      float* IREE_UK_RESTRICT out_ptr = out_tile;
      __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
      __riscv_vse32_v_f32m4(out_ptr + N0, acc1, vl);
    } else {
      _Float16* IREE_UK_RESTRICT out_ptr = out_tile;
      __riscv_vse16_v_f16m2(out_ptr, __riscv_vfncvt_f_f_w_f16m2(acc0, vl), vl);
      __riscv_vse16_v_f16m2(out_ptr + N0, __riscv_vfncvt_f_f_w_f16m2(acc1, vl),
                            vl);
    }
  } else if (M0 == 4) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      if (acc_type == IREE_UK_TYPE_FLOAT_32) {
        float* IREE_UK_RESTRICT out_ptr = out_tile;
        acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
        acc1 = __riscv_vle32_v_f32m4(out_ptr + N0, vl);
        acc2 = __riscv_vle32_v_f32m4(out_ptr + N0 * 2, vl);
        acc3 = __riscv_vle32_v_f32m4(out_ptr + N0 * 3, vl);
      } else {
        _Float16* IREE_UK_RESTRICT out_ptr = out_tile;
        acc0 =
            __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(out_ptr, vl), vl);
        acc1 = __riscv_vfwcvt_f_f_v_f32m4(
            __riscv_vle16_v_f16m2(out_ptr + N0, vl), vl);
        acc2 = __riscv_vfwcvt_f_f_v_f32m4(
            __riscv_vle16_v_f16m2(out_ptr + N0 * 2, vl), vl);
        acc3 = __riscv_vfwcvt_f_f_v_f32m4(
            __riscv_vle16_v_f16m2(out_ptr + N0 * 3, vl), vl);
      }
    } else {
      acc0 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc1 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc2 = __riscv_vfmv_v_f_f32m4(0.0, vl);
      acc3 = __riscv_vfmv_v_f_f32m4(0.0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat16m2_t rhs = __riscv_vle16_v_f16m2(rhs_ptr, vl);

      rhs_ptr += N0;
      _Float16 lhs0 = *lhs_ptr++;
      _Float16 lhs1 = *lhs_ptr++;
      _Float16 lhs2 = *lhs_ptr++;
      _Float16 lhs3 = *lhs_ptr++;

      acc0 = __riscv_vfwmacc_vf_f32m4(acc0, lhs0, rhs, vl);
      acc1 = __riscv_vfwmacc_vf_f32m4(acc1, lhs1, rhs, vl);
      acc2 = __riscv_vfwmacc_vf_f32m4(acc2, lhs2, rhs, vl);
      acc3 = __riscv_vfwmacc_vf_f32m4(acc3, lhs3, rhs, vl);
    }
    if (acc_type == IREE_UK_TYPE_FLOAT_32) {
      float* IREE_UK_RESTRICT out_ptr = out_tile;
      __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
      __riscv_vse32_v_f32m4(out_ptr + N0, acc1, vl);
      __riscv_vse32_v_f32m4(out_ptr + N0 * 2, acc2, vl);
      __riscv_vse32_v_f32m4(out_ptr + N0 * 3, acc3, vl);
    } else {
      _Float16* IREE_UK_RESTRICT out_ptr = out_tile;
      __riscv_vse16_v_f16m2(out_ptr, __riscv_vfncvt_f_f_w_f16m2(acc0, vl), vl);
      __riscv_vse16_v_f16m2(out_ptr + N0, __riscv_vfncvt_f_f_w_f16m2(acc1, vl),
                            vl);
      __riscv_vse16_v_f16m2(out_ptr + N0 * 2,
                            __riscv_vfncvt_f_f_w_f16m2(acc2, vl), vl);
      __riscv_vse16_v_f16m2(out_ptr + N0 * 3,
                            __riscv_vfncvt_f_f_w_f16m2(acc3, vl), vl);
    }
  } else if (M0 == 7) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      if (acc_type == IREE_UK_TYPE_FLOAT_32) {
        float* IREE_UK_RESTRICT out_ptr = out_tile;
        acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
        acc1 = __riscv_vle32_v_f32m4(out_ptr + N0, vl);
        acc2 = __riscv_vle32_v_f32m4(out_ptr + N0 * 2, vl);
        acc3 = __riscv_vle32_v_f32m4(out_ptr + N0 * 3, vl);
        acc4 = __riscv_vle32_v_f32m4(out_ptr + N0 * 4, vl);
        acc5 = __riscv_vle32_v_f32m4(out_ptr + N0 * 5, vl);
        acc6 = __riscv_vle32_v_f32m4(out_ptr + N0 * 6, vl);
      } else {
        _Float16* IREE_UK_RESTRICT out_ptr = out_tile;
        acc0 =
            __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(out_ptr, vl), vl);
        acc1 = __riscv_vfwcvt_f_f_v_f32m4(
            __riscv_vle16_v_f16m2(out_ptr + N0, vl), vl);
        acc2 = __riscv_vfwcvt_f_f_v_f32m4(
            __riscv_vle16_v_f16m2(out_ptr + N0 * 2, vl), vl);
        acc3 = __riscv_vfwcvt_f_f_v_f32m4(
            __riscv_vle16_v_f16m2(out_ptr + N0 * 3, vl), vl);
        acc4 = __riscv_vfwcvt_f_f_v_f32m4(
            __riscv_vle16_v_f16m2(out_ptr + N0 * 4, vl), vl);
        acc5 = __riscv_vfwcvt_f_f_v_f32m4(
            __riscv_vle16_v_f16m2(out_ptr + N0 * 5, vl), vl);
        acc6 = __riscv_vfwcvt_f_f_v_f32m4(
            __riscv_vle16_v_f16m2(out_ptr + N0 * 6, vl), vl);
      }
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
      vfloat16m2_t rhs = __riscv_vle16_v_f16m2(rhs_ptr, vl);

      rhs_ptr += N0;
      _Float16 lhs0 = *lhs_ptr++;
      _Float16 lhs1 = *lhs_ptr++;
      _Float16 lhs2 = *lhs_ptr++;
      _Float16 lhs3 = *lhs_ptr++;
      _Float16 lhs4 = *lhs_ptr++;
      _Float16 lhs5 = *lhs_ptr++;
      _Float16 lhs6 = *lhs_ptr++;

      acc0 = __riscv_vfwmacc_vf_f32m4(acc0, lhs0, rhs, vl);
      acc1 = __riscv_vfwmacc_vf_f32m4(acc1, lhs1, rhs, vl);
      acc2 = __riscv_vfwmacc_vf_f32m4(acc2, lhs2, rhs, vl);
      acc3 = __riscv_vfwmacc_vf_f32m4(acc3, lhs3, rhs, vl);
      acc4 = __riscv_vfwmacc_vf_f32m4(acc4, lhs4, rhs, vl);
      acc5 = __riscv_vfwmacc_vf_f32m4(acc5, lhs5, rhs, vl);
      acc6 = __riscv_vfwmacc_vf_f32m4(acc6, lhs6, rhs, vl);
    }
    if (acc_type == IREE_UK_TYPE_FLOAT_32) {
      float* IREE_UK_RESTRICT out_ptr = out_tile;
      __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
      __riscv_vse32_v_f32m4(out_ptr + N0, acc1, vl);
      __riscv_vse32_v_f32m4(out_ptr + N0 * 2, acc2, vl);
      __riscv_vse32_v_f32m4(out_ptr + N0 * 3, acc3, vl);
      __riscv_vse32_v_f32m4(out_ptr + N0 * 4, acc4, vl);
      __riscv_vse32_v_f32m4(out_ptr + N0 * 5, acc5, vl);
      __riscv_vse32_v_f32m4(out_ptr + N0 * 6, acc6, vl);
    } else {
      _Float16* IREE_UK_RESTRICT out_ptr = out_tile;
      __riscv_vse16_v_f16m2(out_ptr, __riscv_vfncvt_f_f_w_f16m2(acc0, vl), vl);
      __riscv_vse16_v_f16m2(out_ptr + N0, __riscv_vfncvt_f_f_w_f16m2(acc1, vl),
                            vl);
      __riscv_vse16_v_f16m2(out_ptr + N0 * 2,
                            __riscv_vfncvt_f_f_w_f16m2(acc2, vl), vl);
      __riscv_vse16_v_f16m2(out_ptr + N0 * 3,
                            __riscv_vfncvt_f_f_w_f16m2(acc3, vl), vl);
      __riscv_vse16_v_f16m2(out_ptr + N0 * 4,
                            __riscv_vfncvt_f_f_w_f16m2(acc4, vl), vl);
      __riscv_vse16_v_f16m2(out_ptr + N0 * 5,
                            __riscv_vfncvt_f_f_w_f16m2(acc5, vl), vl);
      __riscv_vse16_v_f16m2(out_ptr + N0 * 6,
                            __riscv_vfncvt_f_f_w_f16m2(acc6, vl), vl);
    }
  } else if (M0 == 8) {
    // Spacemit IME Extension: 8x8x8 Tiling
    // This kernel assumes data has been packed correctly (KxN for RHS).
    
    _Float16* out_ptr = out_tile;
    int K = params->K;

    // 1. Initialize Accumulators (v8-v11)
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
        // Prototype: Zero-init for simplicity. 
        // Strict accumulation would require strided loads of existing data here.
        __asm__ volatile(
            "vsetvli zero, zero, e16, m1 \n\t"
            "vmv.v.i v8, 0 \n\t" "vmv.v.i v9, 0 \n\t"
            "vmv.v.i v10, 0 \n\t" "vmv.v.i v11, 0 \n\t"
            ::: "v8", "v9", "v10", "v11"
        );
    } else {
        __asm__ volatile(
            "vsetvli zero, zero, e16, m1 \n\t"
            "vmv.v.i v8, 0 \n\t" "vmv.v.i v9, 0 \n\t"
            "vmv.v.i v10, 0 \n\t" "vmv.v.i v11, 0 \n\t"
            ::: "v8", "v9", "v10", "v11"
        );
    }

    // 2. Compute Loop (K+=8)
    for (; K > 0; K -= 8) {
        __asm__ volatile(
          "vsetvli zero, zero, e16, m2       \n\t"
          "vle16.v v0, (%[lhs])              \n\t" // LHS Top (4x8)
          "addi    t1, %[lhs], 64            \n\t"
          "vle16.v v2, (t1)                  \n\t" // LHS Bot (4x8)
          "vle16.v v4, (%[rhs])              \n\t" // RHS Left (8x4, Transposed)
          "addi    t1, %[rhs], 64            \n\t"
          "vle16.v v6, (t1)                  \n\t" // RHS Right (8x4, Transposed)
          
          "vsetvli zero, zero, e16, m1       \n\t" 
          "vfmadot v8,  v0, v4               \n\t" // TL
          "vfmadot v9,  v0, v6               \n\t" // TR
          "vfmadot v10, v2, v4               \n\t" // BL
          "vfmadot v11, v2, v6               \n\t" // BR
          : 
          : [lhs] "r"(lhs_ptr), [rhs] "r"(rhs_ptr)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", 
            "v8", "v9", "v10", "v11", "t1", "memory"
        );
        lhs_ptr += 64; 
        rhs_ptr += 64; 
    }

    // 3. Store Results (Scatter 4x4 blocks to 8x8 tile)
    ptrdiff_t stride = 16; // 8 * 2 bytes
    __asm__ volatile(
        "vsetvli zero, zero, e16, m1       \n\t"
        "mv      t1, %[ptr]                \n\t"
        "li      t2, 4                     \n\t"
        "1:                                \n\t" // Top Half
        "vsetvli zero, zero, e16, mf2      \n\t"
        "vse16.v v8, (t1)                  \n\t"
        "addi    t3, t1, 8                 \n\t"
        "vse16.v v9, (t3)                  \n\t"
        "vsetvli zero, zero, e16, m1       \n\t"
        "vslidedown.vi v8, v8, 4           \n\t"
        "vslidedown.vi v9, v9, 4           \n\t"
        "add     t1, t1, %[str]            \n\t"
        "addi    t2, t2, -1                \n\t"
        "bnez    t2, 1b                    \n\t"
        
        "li      t2, 4                     \n\t"
        "2:                                \n\t" // Bot Half
        "vsetvli zero, zero, e16, mf2      \n\t"
        "vse16.v v10, (t1)                 \n\t"
        "addi    t3, t1, 8                 \n\t"
        "vse16.v v11, (t3)                 \n\t"
        "vsetvli zero, zero, e16, m1       \n\t"
        "vslidedown.vi v10, v10, 4         \n\t"
        "vslidedown.vi v11, v11, 4         \n\t"
        "add     t1, t1, %[str]            \n\t"
        "addi    t2, t2, -1                \n\t"
        "bnez    t2, 2b                    \n\t"
        : 
        : [ptr] "r"(out_ptr), [str] "r"(stride)
        : "t1", "t2", "t3", "memory"
    );
  }
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f16f16f16_1xXXx1_to_7xXXx1_riscv_64_zvfh_noskipround(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 7);
  const _Float16* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const _Float16* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  _Float16* out_ptr = out_tile;

  vfloat16m2_t acc0, acc1, acc2, acc3, acc4, acc5, acc6;
  int N0 = params->N0;
  size_t vl = N0;

  if (M0 == 1) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle16_v_f16m2(out_ptr, vl);
    } else {
      acc0 = __riscv_vfmv_v_f_f16m2(0.0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat16m2_t rhs = __riscv_vle16_v_f16m2(rhs_ptr, vl);
      rhs_ptr += N0;
      _Float16 lhs0 = *lhs_ptr++;

      acc0 = __riscv_vfmacc_vf_f16m2(acc0, lhs0, rhs, vl);
    }
    __riscv_vse16_v_f16m2(out_ptr, acc0, vl);
  } else if (M0 == 2) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle16_v_f16m2(out_ptr, vl);
      acc1 = __riscv_vle16_v_f16m2(out_ptr + N0, vl);
    } else {
      acc0 = __riscv_vfmv_v_f_f16m2(0.0, vl);
      acc1 = __riscv_vfmv_v_f_f16m2(0.0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat16m2_t rhs = __riscv_vle16_v_f16m2(rhs_ptr, vl);
      rhs_ptr += N0;
      _Float16 lhs0 = *lhs_ptr++;
      _Float16 lhs1 = *lhs_ptr++;

      acc0 = __riscv_vfmacc_vf_f16m2(acc0, lhs0, rhs, vl);
      acc1 = __riscv_vfmacc_vf_f16m2(acc1, lhs1, rhs, vl);
    }
    __riscv_vse16_v_f16m2(out_ptr, acc0, vl);
    __riscv_vse16_v_f16m2(out_ptr + N0, acc1, vl);
  } else if (M0 == 4) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle16_v_f16m2(out_ptr, vl);
      acc1 = __riscv_vle16_v_f16m2(out_ptr + N0, vl);
      acc2 = __riscv_vle16_v_f16m2(out_ptr + N0 * 2, vl);
      acc3 = __riscv_vle16_v_f16m2(out_ptr + N0 * 3, vl);
    } else {
      acc0 = __riscv_vfmv_v_f_f16m2(0.0, vl);
      acc1 = __riscv_vfmv_v_f_f16m2(0.0, vl);
      acc2 = __riscv_vfmv_v_f_f16m2(0.0, vl);
      acc3 = __riscv_vfmv_v_f_f16m2(0.0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat16m2_t rhs = __riscv_vle16_v_f16m2(rhs_ptr, vl);

      rhs_ptr += N0;
      _Float16 lhs0 = *lhs_ptr++;
      _Float16 lhs1 = *lhs_ptr++;
      _Float16 lhs2 = *lhs_ptr++;
      _Float16 lhs3 = *lhs_ptr++;

      acc0 = __riscv_vfmacc_vf_f16m2(acc0, lhs0, rhs, vl);
      acc1 = __riscv_vfmacc_vf_f16m2(acc1, lhs1, rhs, vl);
      acc2 = __riscv_vfmacc_vf_f16m2(acc2, lhs2, rhs, vl);
      acc3 = __riscv_vfmacc_vf_f16m2(acc3, lhs3, rhs, vl);
    }
    __riscv_vse16_v_f16m2(out_ptr, acc0, vl);
    __riscv_vse16_v_f16m2(out_ptr + N0, acc1, vl);
    __riscv_vse16_v_f16m2(out_ptr + N0 * 2, acc2, vl);
    __riscv_vse16_v_f16m2(out_ptr + N0 * 3, acc3, vl);
  } else if (M0 == 7) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 = __riscv_vle16_v_f16m2(out_ptr, vl);
      acc1 = __riscv_vle16_v_f16m2(out_ptr + N0, vl);
      acc2 = __riscv_vle16_v_f16m2(out_ptr + N0 * 2, vl);
      acc3 = __riscv_vle16_v_f16m2(out_ptr + N0 * 3, vl);
      acc4 = __riscv_vle16_v_f16m2(out_ptr + N0 * 4, vl);
      acc5 = __riscv_vle16_v_f16m2(out_ptr + N0 * 5, vl);
      acc6 = __riscv_vle16_v_f16m2(out_ptr + N0 * 6, vl);
    } else {
      acc0 = __riscv_vfmv_v_f_f16m2(0.0, vl);
      acc1 = __riscv_vfmv_v_f_f16m2(0.0, vl);
      acc2 = __riscv_vfmv_v_f_f16m2(0.0, vl);
      acc3 = __riscv_vfmv_v_f_f16m2(0.0, vl);
      acc4 = __riscv_vfmv_v_f_f16m2(0.0, vl);
      acc5 = __riscv_vfmv_v_f_f16m2(0.0, vl);
      acc6 = __riscv_vfmv_v_f_f16m2(0.0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat16m2_t rhs = __riscv_vle16_v_f16m2(rhs_ptr, vl);

      rhs_ptr += N0;
      _Float16 lhs0 = *lhs_ptr++;
      _Float16 lhs1 = *lhs_ptr++;
      _Float16 lhs2 = *lhs_ptr++;
      _Float16 lhs3 = *lhs_ptr++;
      _Float16 lhs4 = *lhs_ptr++;
      _Float16 lhs5 = *lhs_ptr++;
      _Float16 lhs6 = *lhs_ptr++;

      acc0 = __riscv_vfmacc_vf_f16m2(acc0, lhs0, rhs, vl);
      acc1 = __riscv_vfmacc_vf_f16m2(acc1, lhs1, rhs, vl);
      acc2 = __riscv_vfmacc_vf_f16m2(acc2, lhs2, rhs, vl);
      acc3 = __riscv_vfmacc_vf_f16m2(acc3, lhs3, rhs, vl);
      acc4 = __riscv_vfmacc_vf_f16m2(acc4, lhs4, rhs, vl);
      acc5 = __riscv_vfmacc_vf_f16m2(acc5, lhs5, rhs, vl);
      acc6 = __riscv_vfmacc_vf_f16m2(acc6, lhs6, rhs, vl);
    }
    __riscv_vse16_v_f16m2(out_ptr, acc0, vl);
    __riscv_vse16_v_f16m2(out_ptr + N0, acc1, vl);
    __riscv_vse16_v_f16m2(out_ptr + N0 * 2, acc2, vl);
    __riscv_vse16_v_f16m2(out_ptr + N0 * 3, acc3, vl);
    __riscv_vse16_v_f16m2(out_ptr + N0 * 4, acc4, vl);
    __riscv_vse16_v_f16m2(out_ptr + N0 * 5, acc5, vl);
    __riscv_vse16_v_f16m2(out_ptr + N0 * 6, acc6, vl);
  }
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f16f16f16_1xXXx1_to_7xXXx1_riscv_64_zvfh_skipround(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_f16f16fXX_1xXXx1_to_7xXXx1_riscv_64_zvfh(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_FLOAT_16, M0);
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f16f16f16_1xXXx1_to_7xXXx1_riscv_64_zvfh(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  if (params->flags & IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS) {
    iree_uk_mmt4d_tile_f16f16f16_1xXXx1_to_7xXXx1_riscv_64_zvfh_skipround(
        out_tile, lhs_panel, rhs_panel, params, M0);
  } else {
    iree_uk_mmt4d_tile_f16f16f16_1xXXx1_to_7xXXx1_riscv_64_zvfh_noskipround(
        out_tile, lhs_panel, rhs_panel, params, M0);
  }
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f16f16f32_1xXXx1_to_7xXXx1_riscv_64_zvfh(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_f16f16fXX_1xXXx1_to_7xXXx1_riscv_64_zvfh(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_FLOAT_32, M0);
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f32_1xXXx1_to_7xXXx1_riscv_64_zvfh,
    iree_uk_mmt4d_tile_f16f16f32_1xXXx1_riscv_64_zvfh, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f32_1xXXx1_to_7xXXx1_riscv_64_zvfh,
    iree_uk_mmt4d_tile_f16f16f32_2xXXx1_riscv_64_zvfh, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f32_1xXXx1_to_7xXXx1_riscv_64_zvfh,
    iree_uk_mmt4d_tile_f16f16f32_4xXXx1_riscv_64_zvfh, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f32_1xXXx1_to_7xXXx1_riscv_64_zvfh,
    iree_uk_mmt4d_tile_f16f16f32_7xXXx1_riscv_64_zvfh, 7)

// Register M0=8
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f32_1xXXx1_to_7xXXx1_riscv_64_zvfh,
    iree_uk_mmt4d_tile_f16f16f32_8xXXx1_riscv_64_zvfh, 8)

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f16_1xXXx1_to_7xXXx1_riscv_64_zvfh,
    iree_uk_mmt4d_tile_f16f16f16_1xXXx1_riscv_64_zvfh, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f16_1xXXx1_to_7xXXx1_riscv_64_zvfh,
    iree_uk_mmt4d_tile_f16f16f16_2xXXx1_riscv_64_zvfh, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f16_1xXXx1_to_7xXXx1_riscv_64_zvfh,
    iree_uk_mmt4d_tile_f16f16f16_4xXXx1_riscv_64_zvfh, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f16f16f16_1xXXx1_to_7xXXx1_riscv_64_zvfh,
    iree_uk_mmt4d_tile_f16f16f16_7xXXx1_riscv_64_zvfh, 7)

// Register M0=8
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
  iree_uk_mmt4d_tile_f16f16f16_1xXXx1_to_7xXXx1_riscv_64_zvfh,
  iree_uk_mmt4d_tile_f16f16f16_8xXXx1_riscv_64_zvfh, 8)
