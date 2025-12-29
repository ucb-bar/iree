#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

// --- 1. Fix for __clear_cache ---
// Clang generates calls to this for IREE's instruction cache flushing.
// On RISC-V, this corresponds to the fence.i instruction.
void __clear_cache(void* start, void* end) {
    // RISC-V instruction cache flush
    __asm__ volatile ("fence.i" ::: "memory");
}

// --- 2. Fix for memalign ---
// Newlib sometimes lacks memalign. We map it to memalign if available, 
// or a simple malloc wrapper (warning: alignment might be loose depending on your malloc).
extern void* malloc(size_t size);
extern void free(void* ptr);

void* memalign(size_t alignment, size_t size) {
    // In a strict bare-metal setup, linking against a basic malloc 
    // often yields 8 or 16 byte alignment. 
    // If you need strict alignment, you need a smarter allocator.
    // For now, this satisfies the linker.
    return malloc(size);
}

// --- 3. Fix for _ctype_ (Linker mismatch) ---
// If your headers (Clang) and library (Newlib) disagree on ctype macros.
const char _ctype_[1 + 256] = {0};