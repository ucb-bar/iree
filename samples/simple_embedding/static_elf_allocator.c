// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "samples/simple_embedding/static_elf_allocator.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/allocator.h"

// Static buffer for ELF module storage.
// Size should be large enough for your largest ELF + alignment.
// This can be adjusted based on your needs or made configurable.
#define STATIC_ELF_BUFFER_SIZE (1024 * 1024)  // 1MB
#define STATIC_ELF_BUFFER_ALIGN 4096           // Page alignment

// Statically allocated, aligned buffer for ELF modules.
// Placed in dedicated linker section for deterministic address.
// Use 'used' attribute to prevent optimization, and ensure it's placed in the section.
static uint8_t static_elf_buffer[STATIC_ELF_BUFFER_SIZE]
    __attribute__((section(".iree_elf_buffer"), aligned(STATIC_ELF_BUFFER_ALIGN), used));

// Allocator state tracking.
typedef struct {
  uint8_t* buffer;
  iree_host_size_t capacity;
  bool elf_buffer_allocated;  // Track if ELF buffer is in use
  iree_allocator_t fallback_allocator;  // For small allocations (loader/executable structures)
  bool initialized;  // Track if fallback allocator has been initialized
} static_elf_allocator_state_t;

static static_elf_allocator_state_t allocator_state = {
    .buffer = static_elf_buffer,
    .capacity = STATIC_ELF_BUFFER_SIZE,
    .elf_buffer_allocated = false,
    .fallback_allocator = {NULL, NULL},  // Initialize to null allocator (empty struct)
    .initialized = false,
};

// Threshold for using static buffer vs fallback allocator.
// Small allocations (loader/executable structures) use fallback,
// large allocations (ELF memory) use static buffer.
// ELF allocations are typically page-aligned (4KB+), so we use a lower threshold
// to catch ELF allocations while still using fallback for small structures.
#define STATIC_BUFFER_THRESHOLD (4 * 1024)  // 4KB (one page)

// Custom allocator control function for static ELF buffer.
static iree_status_t static_elf_allocator_ctl(void* self,
                                              iree_allocator_command_t command,
                                              const void* params,
                                              void** inout_ptr) {
  static_elf_allocator_state_t* state = (static_elf_allocator_state_t*)self;

  // Initialize fallback allocator on first use (lazy initialization).
  if (!state->initialized) {
    state->fallback_allocator = iree_allocator_system();
    state->initialized = true;
  }

  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC: {
      const iree_allocator_alloc_params_t* alloc_params =
          (const iree_allocator_alloc_params_t*)params;

      // DEBUG: Print allocation request
      fprintf(stdout, "[DEBUG] Static allocator: requested 0x%x bytes (threshold=0x%x)\n",
              (unsigned int)alloc_params->byte_length, STATIC_BUFFER_THRESHOLD);
      fflush(stdout);

      // For small allocations (loader/executable structures), use fallback.
      // ELF allocations are typically >= 4KB (page-aligned), so we use the
      // static buffer for those. Very small allocations (< 4KB) are likely
      // loader structures and use the fallback.
      if (alloc_params->byte_length < STATIC_BUFFER_THRESHOLD) {
        fprintf(stdout, "[DEBUG] Static allocator: using fallback allocator (size < threshold)\n");
        fflush(stdout);
        return iree_allocator_malloc(state->fallback_allocator,
                                     alloc_params->byte_length, inout_ptr);
      }

      // For large allocations (ELF memory), use static buffer.
      // Check if buffer is already allocated.
      if (state->elf_buffer_allocated) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "Static ELF buffer already allocated (only one ELF supported)");
      }

      // Check if requested size fits in buffer.
      if (alloc_params->byte_length > state->capacity) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "ELF size %" PRIhsz
            " exceeds static buffer capacity %" PRIhsz,
            alloc_params->byte_length, state->capacity);
      }

      // Zero memory if CALLOC.
      if (command == IREE_ALLOCATOR_COMMAND_CALLOC) {
        memset(state->buffer, 0, alloc_params->byte_length);
      }

      // Return pointer to static buffer.
      *inout_ptr = state->buffer;
      state->elf_buffer_allocated = true;
      fprintf(stdout, "[DEBUG] Static allocator: using static buffer at %p\n", state->buffer);
      fflush(stdout);
      return iree_ok_status();
    }

    case IREE_ALLOCATOR_COMMAND_FREE: {
      // Check if this is the static buffer.
      if (*inout_ptr == state->buffer) {
        state->elf_buffer_allocated = false;
        return iree_ok_status();
      }
      // Otherwise, free using fallback allocator.
      iree_allocator_free(state->fallback_allocator, *inout_ptr);
      return iree_ok_status();
    }

    case IREE_ALLOCATOR_COMMAND_REALLOC: {
      // Realloc not supported for static buffer.
      // For small allocations, try fallback.
      // This is a simplified implementation - in practice, realloc is rarely
      // used for ELF loading.
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Realloc not supported for static ELF allocator");
    }

    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Unknown allocator command");
  }
}

iree_allocator_t iree_samples_static_elf_allocator(void) {
  // DEBUG: Print static buffer address to verify it's in the expected section
  // The buffer should be placed at ~0x80068000 based on the linker script section .iree_elf_buffer
  void* buffer_addr = allocator_state.buffer;
  
  fprintf(stdout, "[DEBUG] Static allocator: static buffer at %p (expected ~0x80068000)\n", buffer_addr);
  fprintf(stdout, "[DEBUG] Static allocator: buffer size = 0x%x bytes\n",
          (unsigned int)allocator_state.capacity);
  
  // The buffer is placed in the .iree_elf_buffer section via __attribute__((section(".iree_elf_buffer")))
  // This ensures it's at the deterministic address specified in the linker script
  fprintf(stdout, "[DEBUG] Static allocator: buffer placed via __attribute__((section(\".iree_elf_buffer\")))\n");
  fflush(stdout);
  
  iree_allocator_t allocator = {
      .self = &allocator_state,
      .ctl = static_elf_allocator_ctl,
  };
  return allocator;
}

void* iree_samples_get_static_elf_base_address(void) {
  return allocator_state.buffer;
}

iree_host_size_t iree_samples_get_static_elf_buffer_size(void) {
  return allocator_state.capacity;
}

bool iree_samples_elf_is_in_static_buffer(const void* elf_data,
                                           iree_host_size_t elf_size) {
  if (!elf_data || elf_size == 0) {
    return false;
  }
  
  uint8_t* buffer_start = allocator_state.buffer;
  uint8_t* buffer_end = buffer_start + allocator_state.capacity;
  const uint8_t* elf_start = (const uint8_t*)elf_data;
  const uint8_t* elf_end = elf_start + elf_size;
  
  // Check if ELF data is entirely within the static buffer
  return (elf_start >= buffer_start) && (elf_end <= buffer_end);
}

void* iree_samples_get_static_elf_vaddr_bias(void) {
  // In baremetal with 1:1 physical addressing, the static buffer base
  // can be used directly as vaddr_bias (with potential small offset for
  // ELF's virtual address base, typically 0 or small alignment).
  // 
  // For true 1:1 mapping, this would be:
  //   vaddr_bias = static_buffer_base - elf_vaddr_min
  // But if ELF vaddrs start at 0 or match physical addresses, it's just:
  //   vaddr_bias = static_buffer_base
  return allocator_state.buffer;
}

