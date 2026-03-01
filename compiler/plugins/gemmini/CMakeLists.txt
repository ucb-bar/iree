iree_add_all_subdirs()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/iree/compiler/ThirdParty/buddy_gemmini
                 src/iree/compiler/ThirdParty/buddy_gemmini)

iree_compiler_register_plugin(
  PLUGIN_ID
    gemmini
  TARGET
    ::registration
)

iree_cc_library(
  NAME
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    BuddyGemminiRegistration
    MLIRIR
    iree::compiler::PluginAPI
  PUBLIC
)

iree_lit_test_suite(
  NAME
    gemmini_matmul_lowering_test
  SRCS
    "test/gemmini_matmul_lowering.mlir"
  TOOLS
    FileCheck
    iree-compile
)
