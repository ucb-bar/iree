// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/local/elf/arch.h"
#include "iree/hal/local/elf/elf_types.h"

#if defined(IREE_ARCH_RISCV_32) || defined(IREE_ARCH_RISCV_64)

// Documentation:
// https://github.com/riscv/riscv-elf-psabi-doc/blob/master/riscv-elf.md

//==============================================================================
// ELF machine type/ABI
//==============================================================================

bool iree_elf_machine_is_valid(iree_elf_half_t machine) {
  return machine == 0xF3;  // EM_RISCV / 243
}

//==============================================================================
// ELF relocations
//==============================================================================

enum {
  IREE_ELF_R_RISCV_NONE = 0,
  IREE_ELF_R_RISCV_32 = 1,
  IREE_ELF_R_RISCV_64 = 2,
  IREE_ELF_R_RISCV_RELATIVE = 3,
  IREE_ELF_R_RISCV_COPY = 4,
  IREE_ELF_R_RISCV_JUMP_SLOT = 5,
};

#if defined(IREE_ARCH_RISCV_32)
static iree_status_t iree_elf_arch_riscv_apply_rela(
    iree_elf_relocation_state_t* state, iree_host_size_t rela_count,
    const iree_elf_rela_t* rela_table) {
  for (iree_host_size_t i = 0; i < rela_count; ++i) {
    const iree_elf_rela_t* rela = &rela_table[i];
    uint32_t type = IREE_ELF_R_TYPE(rela->r_info);
    if (type == 0) continue;

    iree_elf_addr_t sym_addr = 0;
    uint32_t sym_ordinal = (uint32_t)IREE_ELF_R_SYM(rela->r_info);
    if (sym_ordinal != 0) {
      if (sym_ordinal >= state->dynsym_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "invalid symbol in relocation: %u",
                                sym_ordinal);
      }
      sym_addr = (iree_elf_addr_t)state->vaddr_bias +
                 state->dynsym[sym_ordinal].st_value;
    }

    iree_elf_addr_t instr_ptr =
        (iree_elf_addr_t)state->vaddr_bias + rela->r_offset;
    switch (type) {
      case IREE_ELF_R_RISCV_NONE:
        break;
      case IREE_ELF_R_RISCV_32:
        *(uint32_t*)instr_ptr = (uint32_t)(sym_addr + rela->r_addend);
        break;
      case IREE_ELF_R_RISCV_JUMP_SLOT:
        *(uint32_t*)instr_ptr = (uint32_t)sym_addr;
        break;
      case IREE_ELF_R_RISCV_RELATIVE:
        *(uint32_t*)instr_ptr = (uint32_t)(state->vaddr_bias + rela->r_addend);
        break;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unimplemented riscv32 relocation type %08X",
                                type);
    }
  }
  return iree_ok_status();
}
#else   // IREE_ARCH_RISCV_64
static iree_status_t iree_elf_arch_riscv_apply_rela(
    iree_elf_relocation_state_t* state, iree_host_size_t rela_count,
    const iree_elf_rela_t* rela_table) {
  for (iree_host_size_t i = 0; i < rela_count; ++i) {
    const iree_elf_rela_t* rela = &rela_table[i];
    uint32_t type = IREE_ELF_R_TYPE(rela->r_info);
    if (type == 0) continue;

    iree_elf_addr_t sym_addr = 0;
    uint32_t sym_ordinal = (uint32_t)IREE_ELF_R_SYM(rela->r_info);
    if (sym_ordinal != 0) {
      if (sym_ordinal >= state->dynsym_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "invalid symbol in relocation: %u",
                                sym_ordinal);
      }
      sym_addr = (iree_elf_addr_t)state->vaddr_bias +
                 state->dynsym[sym_ordinal].st_value;
    }

    iree_elf_addr_t instr_ptr =
        (iree_elf_addr_t)state->vaddr_bias + rela->r_offset;
    switch (type) {
      case IREE_ELF_R_RISCV_NONE:
        break;
      case IREE_ELF_R_RISCV_32:
        *(uint32_t*)instr_ptr = (uint32_t)(sym_addr + rela->r_addend);
        break;
      case IREE_ELF_R_RISCV_64:
        *(uint64_t*)instr_ptr = (uint64_t)(sym_addr + rela->r_addend);
        break;
      case IREE_ELF_R_RISCV_JUMP_SLOT:
        *(uint64_t*)instr_ptr = (uint64_t)sym_addr;
        break;
      case IREE_ELF_R_RISCV_RELATIVE:
        *(uint64_t*)instr_ptr = (uint64_t)(state->vaddr_bias + rela->r_addend);
        break;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unimplemented riscv64 relocation type %08X",
                                type);
    }
  }
  return iree_ok_status();
}
#endif  // IREE_ARCH_RISCV_*

iree_status_t iree_elf_arch_apply_relocations(
    iree_elf_relocation_state_t* state) {
  // Gather the relevant relocation tables.
  iree_host_size_t rela_count = 0;
  const iree_elf_rela_t* rela_table = NULL;
  for (iree_host_size_t i = 0; i < state->dyn_table_count; ++i) {
    const iree_elf_dyn_t* dyn = &state->dyn_table[i];
    switch (dyn->d_tag) {
      case IREE_ELF_DT_RELA:
        rela_table =
            (const iree_elf_rela_t*)(state->vaddr_bias + dyn->d_un.d_ptr);
        break;
      case IREE_ELF_DT_RELASZ:
        rela_count = dyn->d_un.d_val / sizeof(iree_elf_rela_t);
        break;

      case IREE_ELF_DT_REL:
      case IREE_ELF_DT_RELSZ:
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "unsupported DT_REL relocations");
      default:
        // Ignored.
        break;
    }
  }
  if (!rela_table) rela_count = 0;

  if (rela_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_elf_arch_riscv_apply_rela(state, rela_count, rela_table));
  }

  return iree_ok_status();
}

//==============================================================================
// Cross-ABI function calls
//==============================================================================

// Helper function to print first 32 bytes of function code in hex
static void print_function_bytes(const void* symbol_ptr) {
  const uint8_t* bytes = (const uint8_t*)symbol_ptr;
  fprintf(stdout, "  function bytes (first 32): ");
  for (int i = 0; i < 32; i++) {
    fprintf(stdout, "%02x ", bytes[i]);
  }
  fprintf(stdout, "\n");
}

void iree_elf_call_v_v(const void* symbol_ptr) {
  typedef void (*ptr_t)(void);
  fprintf(stdout, "[DEBUG] iree_elf_call_v_v: calling function at %p\n", symbol_ptr);
  print_function_bytes(symbol_ptr);
  ((ptr_t)symbol_ptr)();
}

void* iree_elf_call_p_i(const void* symbol_ptr, int a0) {
  typedef void* (*ptr_t)(int);
  fprintf(stdout, "[DEBUG] iree_elf_call_p_i: calling function at %p (a0=%d)\n", 
          symbol_ptr, a0);
  print_function_bytes(symbol_ptr);
  void* ret = ((ptr_t)symbol_ptr)(a0);
  fprintf(stdout, "  return: %p\n", ret);
  return ret;
}

void* iree_elf_call_p_ip(const void* symbol_ptr, int a0, void* a1) {
  typedef void* (*ptr_t)(int, void*);
  fprintf(stdout, "[DEBUG] iree_elf_call_p_ip: calling function at %p (a0=%d, a1=%p)\n", 
          symbol_ptr, a0, a1);
  print_function_bytes(symbol_ptr);
  void* ret = ((ptr_t)symbol_ptr)(a0, a1);
  fprintf(stdout, "  return: %p\n", ret);
  return ret;
}

int iree_elf_call_i_p(const void* symbol_ptr, void* a0) {
  typedef int (*ptr_t)(void*);
  fprintf(stdout, "[DEBUG] iree_elf_call_i_p: calling function at %p (a0=%p)\n", 
          symbol_ptr, a0);
  print_function_bytes(symbol_ptr);
  int ret = ((ptr_t)symbol_ptr)(a0);
  fprintf(stdout, "  return: %d (0x%x)\n", ret, ret);
  return ret;
}

int iree_elf_call_i_ppp(const void* symbol_ptr, void* a0, void* a1, void* a2) {
  typedef int (*ptr_t)(void*, void*, void*);
  // DEBUG: Print function address and arguments
  fprintf(stdout, "[DEBUG] iree_elf_call_i_ppp: calling function at %p\n", symbol_ptr);
  fprintf(stdout, "  args: a0=%p, a1=%p, a2=%p\n", a0, a1, a2);
  print_function_bytes(symbol_ptr);
  int ret = ((ptr_t)symbol_ptr)(a0, a1, a2);
  fprintf(stdout, "  return: %d (0x%x)\n", ret, ret);
  return ret;
}

void* iree_elf_call_p_ppp(const void* symbol_ptr, void* a0, void* a1,
                          void* a2) {
  typedef void* (*ptr_t)(void*, void*, void*);
  fprintf(stdout, "[DEBUG] iree_elf_call_p_ppp: calling function at %p\n", symbol_ptr);
  fprintf(stdout, "  args: a0=%p, a1=%p, a2=%p\n", a0, a1, a2);
  print_function_bytes(symbol_ptr);
  void* ret = ((ptr_t)symbol_ptr)(a0, a1, a2);
  fprintf(stdout, "  return: %p\n", ret);
  return ret;
}

int iree_elf_thunk_i_ppp(const void* symbol_ptr, void* a0, void* a1, void* a2) {
  typedef int (*ptr_t)(void*, void*, void*);
  fprintf(stdout, "[DEBUG] iree_elf_thunk_i_ppp: calling function at %p\n", symbol_ptr);
  fprintf(stdout, "  args: a0=%p, a1=%p, a2=%p\n", a0, a1, a2);
  print_function_bytes(symbol_ptr);
  int ret = ((ptr_t)symbol_ptr)(a0, a1, a2);
  fprintf(stdout, "  return: %d (0x%x)\n", ret, ret);
  return ret;
}

#endif  // IREE_ARCH_RISCV_*
