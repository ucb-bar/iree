//===- RegisterGemmini.cpp - Registration for Gemmini dialect & passes ----===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "RegisterGemmini.h"
#include "Gemmini/GemminiDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {

void registerGemminiDialect(DialectRegistry &registry) {
  registry.insert<::buddy::gemmini::GemminiDialect>();
}

void registerGemminiPasses() {
  // Register passes
  mlir::buddy::registerLowerLinalgToGemminiPass();
  mlir::buddy::registerLowerGemminiPass();
  mlir::buddy::registerGemminiIRDumpsPass();

  static PassPipelineRegistration<> gemminiTestPassPipeline(
      "iree-gemmini-test-pipeline",
      "Runs one-shot bufferization and lowers linalg to Gemmini",
      [](OpPassManager &passManager) {
        (void)parsePassPipeline("one-shot-bufferize,convert-linalg-to-gemmini",
                                passManager);
      });
}

} // namespace iree_compiler
} // namespace mlir
