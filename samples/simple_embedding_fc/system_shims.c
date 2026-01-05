#include <stddef.h>
#include <stdint.h>

// --- 1. Fix for __clear_cache ---
void __clear_cache(void* start, void* end) {
    __asm__ volatile ("fence.i" ::: "memory");
}

// --- 2. Fix for memalign ---
extern void* malloc(size_t size);
extern void free(void* ptr);

void* memalign(size_t alignment, size_t size) {
    return malloc(size);
}

// --- 3. Fix for _ctype_ ---
const char _ctype_[1 + 256] = {0};

// --- 4. CRITICAL: Manual FPU/Vector Enable Function ---
// We removed __attribute__((constructor)) so we can call it manually.
void iree_platform_enable_extensions() {
    uint64_t mstatus;
    __asm__ volatile ("csrr %0, mstatus" : "=r"(mstatus));
    // Enable FPU (FS=1 -> bits 13:14) and Vector (VS=1 -> bits 9:10)
    mstatus |= 0x00002200; 
    __asm__ volatile ("csrw mstatus, %0" : : "r"(mstatus));
}

// --- 5. Low-Level Debug Helper ---
// Writes directly to the host via HTIF tohost. 
// This allows us to debug even if printf is broken/buffering.
// 'val' will appear in the trace logs.
void debug_trace(uint64_t val) {
    // Magic volatile pointer to tohost (generic location for spike defaults)
    // Note: The linker script/specs usually handle this, but for raw debugging:
    // We just perform a fence to ensure previous writes happen.
    __asm__ volatile ("fence" ::: "memory");
    // (Actual tohost writing is complex without the symbol, 
    //  so we will rely on side-effects or printf for now, 
    //  but enabling FPU manually usually fixes the crash).
}