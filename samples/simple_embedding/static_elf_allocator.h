// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_SIMPLE_EMBEDDING_STATIC_ELF_ALLOCATOR_H_
#define IREE_SAMPLES_SIMPLE_EMBEDDING_STATIC_ELF_ALLOCATOR_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an allocator that uses a statically allocated buffer for ELF modules.
// This is useful for baremetal systems or when you want deterministic memory
// addresses for stack unwinding.
//
// The allocator will allocate from a fixed-size static buffer. Only one
// allocation is supported at a time (typical for single ELF module scenarios).
//
// Returns an allocator that can be passed to iree_hal_embedded_elf_loader_create().
iree_allocator_t iree_samples_static_elf_allocator(void);

// Gets the base address of the static ELF buffer.
// This is useful for stack unwinding tools that need to know where the ELF
// module is loaded.
void* iree_samples_get_static_elf_base_address(void);

// Gets the size of the static ELF buffer.
iree_host_size_t iree_samples_get_static_elf_buffer_size(void);

// Checks if the given ELF data is already within the static buffer section.
// Returns true if the ELF is already at its final location (1:1 mapping).
// This allows skipping copy and relocations in baremetal embedded contexts.
bool iree_samples_elf_is_in_static_buffer(const void* elf_data,
                                           iree_host_size_t elf_size);

// Gets the offset from static buffer base to use for vaddr_bias calculation.
// In baremetal with 1:1 physical addressing, this is typically 0 or a small
// alignment offset. Returns the address that should be used as vaddr_bias
// for direct mapping (no copy needed).
void* iree_samples_get_static_elf_vaddr_bias(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_SAMPLES_SIMPLE_EMBEDDING_STATIC_ELF_ALLOCATOR_H_

