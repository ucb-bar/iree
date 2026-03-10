################################################################################
# Concurrent Scheduling Example
# Demonstrates concurrent execution of multiple neural network models with
# both vanilla IREE scheduling and custom oracle scheduling.
################################################################################

# Compile the three model MLIR files to VMFB
iree_bytecode_module(
  NAME
    model_a_conv
  SRC
    "model_a_conv.mlir"
  FLAGS
    "--iree-hal-target-device=local"
    "--iree-hal-local-target-device-backends=llvm-cpu"
  PUBLIC
)

iree_bytecode_module(
  NAME
    model_b_dense
  SRC
    "model_b_dense.mlir"
  FLAGS
    "--iree-hal-target-device=local"
    "--iree-hal-local-target-device-backends=llvm-cpu"
  PUBLIC
)

iree_bytecode_module(
  NAME
    model_c_residual
  SRC
    "model_c_residual.mlir"
  FLAGS
    "--iree-hal-target-device=local"
    "--iree-hal-local-target-device-backends=llvm-cpu"
  PUBLIC
)

# Compile the vanilla async pipeline
iree_bytecode_module(
  NAME
    pipeline_vanilla_async
  SRC
    "pipeline_vanilla_async.mlir"
  FLAGS
    "--iree-execution-model=async-external"
    "--iree-hal-target-device=local"
    "--iree-hal-local-target-device-backends=llvm-cpu"
  PUBLIC
)

# Oracle scheduler executable
iree_cc_binary(
  NAME
    concurrent_scheduling_oracle
  SRCS
    "concurrent_scheduling_oracle.c"
  DEPS
    iree::base
    iree::hal
    iree::modules::hal::types
    iree::runtime
)

# Vanilla scheduler executable
iree_cc_binary(
  NAME
    concurrent_scheduling_vanilla
  SRCS
    "concurrent_scheduling_vanilla.c"
  DEPS
    iree::base
    iree::hal
    iree::modules::hal::types
    iree::runtime
)

# Add lit tests for the MLIR pipeline
iree_lit_test_suite(
  NAME
    lit
  SRCS
    "pipeline_vanilla_async.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
  DATA
    model_a_conv.mlir
    model_b_dense.mlir
    model_c_residual.mlir
  LABELS
    "driver=local-task"
    "hostonly"
)
