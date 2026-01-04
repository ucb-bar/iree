// Copyright 2023 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/pack_riscv_64_internal.h"
#include <riscv_vector.h>

// [ADD] The Transposing Packer Implementation
// Fixed: Changed iree_uk_ssize_t -> iree_uk_index_t to match IREE API
static void iree_uk_pack_tile_f16f16f16_8x8_riscv_64_ime_rhs(
    void* IREE_UK_RESTRICT out_tile_ptr, const void* IREE_UK_RESTRICT in_tile_ptr,
    iree_uk_index_t outer_size1, iree_uk_index_t out_stride1,
    iree_uk_index_t in_stride0, iree_uk_index_t elem_size,
    iree_uk_index_t tile_size0, iree_uk_index_t tile_size1) {
    
  const _Float16* src = (const _Float16*)in_tile_ptr;
  _Float16* dst = (_Float16*)out_tile_ptr;
  
  for (iree_uk_index_t i = 0; i < outer_size1; ++i) {
      // Transpose 8x8 block using RVV
      __asm__ volatile(
        "vsetvli zero, zero, e16, m1       \n\t"
        "slli    t1, %[in_str], 1          \n\t" // stride in bytes
        "vlse16.v v0, (%[src]), t1         \n\t"
        "add     %[src], %[src], t1        \n\t"
        "vlse16.v v1, (%[src]), t1         \n\t"
        "add     %[src], %[src], t1        \n\t"
        "vlse16.v v2, (%[src]), t1         \n\t"
        "add     %[src], %[src], t1        \n\t"
        "vlse16.v v3, (%[src]), t1         \n\t"
        "add     %[src], %[src], t1        \n\t"
        "vlse16.v v4, (%[src]), t1         \n\t"
        "add     %[src], %[src], t1        \n\t"
        "vlse16.v v5, (%[src]), t1         \n\t"
        "add     %[src], %[src], t1        \n\t"
        "vlse16.v v6, (%[src]), t1         \n\t"
        "add     %[src], %[src], t1        \n\t"
        "vlse16.v v7, (%[src]), t1         \n\t"
        "li      t2, 16                    \n\t" 
        "vsse16.v v0, (%[dst]), t2         \n\t"
        "addi    %[dst], %[dst], 2         \n\t"
        "vsse16.v v1, (%[dst]), t2         \n\t"
        "addi    %[dst], %[dst], 2         \n\t"
        "vsse16.v v2, (%[dst]), t2         \n\t"
        "addi    %[dst], %[dst], 2         \n\t"
        "vsse16.v v3, (%[dst]), t2         \n\t"
        "addi    %[dst], %[dst], 2         \n\t"
        "vsse16.v v4, (%[dst]), t2         \n\t"
        "addi    %[dst], %[dst], 2         \n\t"
        "vsse16.v v5, (%[dst]), t2         \n\t"
        "addi    %[dst], %[dst], 2         \n\t"
        "vsse16.v v6, (%[dst]), t2         \n\t"
        "addi    %[dst], %[dst], 2         \n\t"
        "vsse16.v v7, (%[dst]), t2         \n\t"
        : 
        : [src] "r"(src), [dst] "r"(dst), [in_str] "r"(in_stride0)
        : "v0","v1","v2","v3","v4","v5","v6","v7", "t1", "t2", "memory"
      );
      // Advance pointers
      src = (const _Float16*)in_tile_ptr + (i + 1) * 8; 
      dst = (_Float16*)out_tile_ptr + (i + 1) * out_stride1;
  }
}

iree_uk_pack_tile_func_t iree_uk_pack_select_tile_func_arch(
    const iree_uk_pack_params_t* params) {
  
  // [ADD] Selector Logic
  if (params->elem_size == 2 && 
      params->tile_size0 == 8 && 
      params->tile_size1 == 8) {
      return iree_uk_pack_tile_f16f16f16_8x8_riscv_64_ime_rhs;
  }

  return 0;
}