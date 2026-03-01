#include "iree/compiler/PluginAPI/Client.h"

#include "compiler/plugins/gemmini/src/iree/compiler/ThirdParty/buddy_gemmini/RegisterGemmini.h"

namespace mlir::iree_compiler {
namespace {

struct GemminiSession
    : public PluginSession<GemminiSession, EmptyPluginOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() { registerGemminiPasses(); }

  void onRegisterDialects(DialectRegistry &registry) override {
    registerGemminiDialect(registry);
  }
};

} // namespace
} // namespace mlir::iree_compiler

extern "C" bool iree_register_compiler_plugin_gemmini(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::GemminiSession>("gemmini");
  return true;
}
