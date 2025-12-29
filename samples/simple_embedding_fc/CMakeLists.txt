# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

# Force RISC-V 64 targeting
iree_compiler_targeting_iree_arch(_IREE_COMPILER_TARGETING_RISCV_64 "riscv_64")

# Only proceed if we have the Local Sync driver and Embedded ELF loader
if(IREE_HAL_DRIVER_LOCAL_SYNC AND
   ((IREE_TARGET_BACKEND_LLVM_CPU AND _IREE_COMPILER_TARGETING_RISCV_64) OR IREE_HOST_BIN_DIR))

  # =========================================================================
  # 1. Standard RISC-V 64 Sync Target (LLVM-CPU)
  # =========================================================================
  if(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
    iree_cc_binary(
      NAME
        simple_embedding_embedded_sync_fc
      SRCS
        "device_embedded_sync.c"
        "simple_embedding.c"
        "system_shims.c"
      DEPS
        ::simple_embedding_test_bytecode_module_cpu_riscv_64_c
        iree::base
        iree::hal
        iree::hal::drivers::local_sync::sync_driver
        iree::hal::local
        iree::hal::local::loaders::embedded_elf_loader
        iree::modules::hal
        iree::vm
        iree::vm::bytecode::module
      LINKOPTS          # <--- ADD THIS SECTION
        "-static"       # <--- ADD THIS FLAG
        "-Wl,--start-group"
        "-lc"
        "-lm"
        "-lgcc"
        "-lgloss_htif"
        "-Wl,--end-group"
      DEFINES             # <--- ADD THIS SECTION
        "IREE_STATUS_FEATURES=0" # Disables complex error strings (removes snprintf deps)
    )

    #set_target_properties(simple_embedding_embedded_sync PROPERTIES LINKER_LANGUAGE C)

    iree_bytecode_module(
      NAME
        simple_embedding_test_bytecode_module_cpu_riscv_64
      SRC
        "model.mlir"
      C_IDENTIFIER
        "iree_samples_simple_embedding_test_module_cpu_riscv_64"
      FLAGS
        "--iree-hal-target-device=local"
        "--iree-hal-local-target-device-backends=llvm-cpu"
        "--iree-llvmcpu-target-triple=riscv64-unknown-elf"
        "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c"
        "--iree-llvmcpu-target-abi=lp64d"
        "--iree-llvmcpu-debug-symbols=false"
        "--iree-vm-bytecode-module-strip-source-map=true"
        "--iree-vm-emit-polyglot-zip=false"
      PUBLIC
    )

    iree_native_test(
      NAME
        "simple_embedding_embedded_sync_test"
      SRC
        ::simple_embedding_embedded_sync_fc
    )
  endif()

  # =========================================================================
  # 2. Vector RISC-V 64 Sync Target (LLVM-CPU Vector)
  # =========================================================================
  if(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
    iree_cc_binary(
      NAME
        simple_embedding_embedded_sync_vector_fc
      SRCS
        "device_embedded_sync.c"
        "simple_embedding.c"
      DEPS
        ::simple_embedding_test_bytecode_module_cpu_riscv_64_vector_c
        iree::base
        iree::hal
        iree::hal::drivers::local_sync::sync_driver
        iree::hal::local
        iree::hal::local::loaders::embedded_elf_loader
        iree::modules::hal
        iree::vm
        iree::vm::bytecode::module
      DEFINES
        "USE_VECTOR_MODULE=1"
      LINKOPTS          # <--- ADD THIS SECTION
        "-static"       # <--- ADD THIS FLAG
    )

    iree_bytecode_module(
      NAME
        simple_embedding_test_bytecode_module_cpu_riscv_64_vector
      SRC
        "model.mlir"
      C_IDENTIFIER
        "iree_samples_simple_embedding_test_module_cpu_riscv_64_vector"
      FLAGS
        "--iree-hal-target-device=local"
        "--iree-hal-local-target-device-backends=llvm-cpu"
        "--iree-llvmcpu-target-triple=riscv64-unknown-elf"
        "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+zvl128b,+zvfh,+zvbb"
        "--iree-llvmcpu-target-abi=lp64d"
        "--iree-llvmcpu-target-vector-width-in-bytes=16"
        "--riscv-v-fixed-length-vector-lmul-max=2"
        "--iree-llvmcpu-debug-symbols=false"
        "--iree-vm-bytecode-module-strip-source-map=true"
        "--iree-vm-emit-polyglot-zip=false"
      PUBLIC
    )

    iree_native_test(
      NAME
        "simple_embedding_embedded_sync_vector_test"
      SRC
        ::simple_embedding_embedded_sync_vector_fc
    )
  endif()

  # =========================================================================
  # 3. VMVX Sync Target (Interpreter)
  # =========================================================================
  if(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE)
    iree_cc_binary(
      NAME
        simple_embedding_embedded_sync_vmvx_fc
      SRCS
        "device_vmvx_sync.c"
        "simple_embedding.c"
      DEPS
        ::simple_embedding_test_bytecode_module_vmvx_c
        iree::base
        iree::hal
        iree::hal::drivers::local_sync::sync_driver
        iree::hal::local
        iree::hal::local::loaders::vmvx_module_loader
        iree::modules::hal
        iree::vm
        iree::vm::bytecode::module
    )

    iree_bytecode_module(
      NAME
        simple_embedding_test_bytecode_module_vmvx
      SRC
        "model.mlir"
      C_IDENTIFIER
        "iree_samples_simple_embedding_test_module_vmvx"
      FLAGS
        "--iree-hal-target-device=local"
        "--iree-hal-local-target-device-backends=vmvx"
        "--iree-vm-bytecode-module-strip-source-map=true"
        "--iree-vm-emit-polyglot-zip=false"
      PUBLIC
    )

    iree_native_test(
      NAME
        "simple_embedding_embedded_sync_vmvx_test"
      SRC
        ::simple_embedding_embedded_sync_vmvx_fc
    )
  endif()

endif()