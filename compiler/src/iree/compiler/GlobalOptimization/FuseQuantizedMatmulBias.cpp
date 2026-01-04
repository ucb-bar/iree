// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h" 
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Included from CPULowerToUKernels.cpp
static bool isInitializedToZero(mlir::Value outsOperand) {
  auto fillOp = outsOperand.getDefiningOp<mlir::linalg::FillOp>();
  if (!fillOp)
    return false;
  mlir::Value fillVal = fillOp.getDpsInputOperand(0)->get();
  return matchPattern(fillVal, mlir::m_Zero()) ||
         matchPattern(fillVal, mlir::m_AnyZeroFloat());
}

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_FUSEQUANTIZEDMATMULBIASPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

// ... (matchDequant and matchQuant helpers remain the same) ...
static std::optional<std::pair<mlir::Value, mlir::Value>>
matchDequant(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      op.getBody()->getOperations().size() != 3) {
    return std::nullopt;
  }
  auto yieldOp = cast<linalg::YieldOp>(op.getBody()->getTerminator());
  auto mulfOp = yieldOp.getOperand(0).getDefiningOp<arith::MulFOp>();
  if (!mulfOp) return std::nullopt;

  auto sitofpOp = mulfOp.getLhs().getDefiningOp<arith::SIToFPOp>();
  if (!sitofpOp) return std::nullopt;
  
  auto sitofpIn = sitofpOp.getIn();
  if (auto blockArg = dyn_cast<BlockArgument>(sitofpIn)) {
    if (blockArg.getArgNumber() != 0 || blockArg.getOwner() != op.getBody()) {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }

  mlir::Value scale = mulfOp.getRhs();
  if (scale == sitofpOp.getResult()) {
    scale = mulfOp.getLhs();
  }

  return std::make_pair(op.getDpsInputOperand(0)->get(), scale);
}

static std::optional<std::pair<mlir::Value, mlir::Value>>
matchQuant(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      op.getBody()->getOperations().size() < 4) {
    return std::nullopt;
  }
  auto yieldOp = cast<linalg::YieldOp>(op.getBody()->getTerminator());
  auto fptosiOp = yieldOp.getOperand(0).getDefiningOp<arith::FPToSIOp>();
  if (!fptosiOp) return std::nullopt;

  mlir::Value current = fptosiOp.getIn();
  while (current.getDefiningOp() && !isa<arith::DivFOp>(current.getDefiningOp())) {
    if (auto roundOp = current.getDefiningOp<math::RoundEvenOp>()) {
      current = roundOp.getOperand(); 
    } else if (auto addOp = current.getDefiningOp<arith::AddFOp>()) {
      current = addOp.getLhs();
    } else if (auto maxOp = current.getDefiningOp<arith::MaximumFOp>()) {
      current = maxOp.getLhs();
    } else if (auto minOp = current.getDefiningOp<arith::MinimumFOp>()) {
      current = minOp.getLhs();
    } else {
      return std::nullopt; // Unrecognized op in the chain
    }
  }
  
  auto divfOp = current.getDefiningOp<arith::DivFOp>();
  if (!divfOp) return std::nullopt;
  
  mlir::Value f32Input = divfOp.getLhs();
  mlir::Value quantScale = divfOp.getRhs();
  
  return std::make_pair(f32Input, quantScale);
}


struct FuseQuantizedMatmulBiasAdd
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp addOp,
                                PatternRewriter &rewriter) const override {
    
    // 1. Match the root: the f32 Add op
    if (!linalg::isElementwise(addOp) || addOp.getNumDpsInputs() != 2 ||
        !isa<arith::AddFOp>(addOp.getBody()->getTerminator()->getOperand(0).getDefiningOp())) {
      // This is not the op we're looking for, fail silently.
      return failure();
    }

    // --- FROM THIS POINT ON, WE EMIT DIAGNOSTICS ---
    // This is a candidate op, so we can now use notifyMatchFailure.
    
    // 2. Match the two Dequantization ops
    auto dequantMatmulOp =
        addOp.getDpsInputOperand(0)->get().getDefiningOp<linalg::GenericOp>();
    auto dequantBiasOp =
        addOp.getDpsInputOperand(1)->get().getDefiningOp<linalg::GenericOp>();

    if (!dequantMatmulOp || !dequantBiasOp) {
       // Try swapping operands
       dequantMatmulOp = addOp.getDpsInputOperand(1)->get().getDefiningOp<linalg::GenericOp>();
       dequantBiasOp = addOp.getDpsInputOperand(0)->get().getDefiningOp<linalg::GenericOp>();
       if (!dequantMatmulOp || !dequantBiasOp) {
         return rewriter.notifyMatchFailure(addOp, "FAIL: add inputs not from linalg.generic");
       }
    }

    auto dequantMatmulMatch = matchDequant(dequantMatmulOp);
    if (!dequantMatmulMatch) {
      return rewriter.notifyMatchFailure(dequantMatmulOp, "FAIL: matmul input is not a valid dequant op (sitofp + mulf)");
    }
    
    auto dequantBiasMatch = matchDequant(dequantBiasOp);
    if (!dequantBiasMatch) {
      return rewriter.notifyMatchFailure(dequantBiasOp, "FAIL: bias input is not a valid dequant op (sitofp + mulf)");
    }

    // 3. Check that scales are identical
    Value dequantScaleMatmul = dequantMatmulMatch->second;
    Value dequantScaleBias = dequantBiasMatch->second;
    
    if (dequantScaleMatmul != dequantScaleBias) {
       return rewriter.notifyMatchFailure(addOp, "FAIL: dequant scales do not match");
    }
    
    // 4. Get the i32 inputs
    Value qgemmResult = dequantMatmulMatch->first;
    Value biasTensor = dequantBiasMatch->first;
    
    // 5. Check that qgemmResult comes from a qgemm with zero-fill
    auto qgemmOp = qgemmResult.getDefiningOp<linalg::QuantizedMatmulOp>();
    if (!qgemmOp) {
       return rewriter.notifyMatchFailure(qgemmResult.getDefiningOp(), "FAIL: matmul result is not a linalg.quantized_matmul");
    }
    
    if (!isInitializedToZero(qgemmOp.getDpsInitOperand(0)->get())) { 
       return rewriter.notifyMatchFailure(qgemmOp, "FAIL: matmul is not from a zero-filled accumulator");
    }
    
    // --- All checks passed. Start rewrite. ---
    Location loc = addOp.getLoc();

    // 6. Create the new fused linalg.quantized_matmul op
    auto newQGemmOp = linalg::QuantizedMatmulOp::create(
        rewriter, loc, qgemmOp.getResult(0).getType(),
        qgemmOp.getDpsInputs(),
        ValueRange{biasTensor});

    // 7. Create the new, single Dequantize op
    IRMapping bvm;
    bvm.map(dequantMatmulOp.getDpsInputOperand(0)->get(), newQGemmOp.getResult(0));
    auto newDequantOp =
        cast<linalg::GenericOp>(rewriter.clone(*dequantMatmulOp, bvm));

    // 8. Replace the original f32 add op
    rewriter.replaceOp(addOp, newDequantOp.getResult(0));
    
    // 9. Clean up the old ops if they are now unused
    rewriter.eraseOp(dequantBiasOp);
    rewriter.eraseOp(dequantMatmulOp);
    rewriter.eraseOp(qgemmOp);

    return success();
  }
};

struct FuseQuantizedMatmulBiasPass
    : public impl::FuseQuantizedMatmulBiasPassBase<
          FuseQuantizedMatmulBiasPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseQuantizedMatmulBiasAdd>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      // We signal failure here to stop the compilation
      // and see the diagnostics.
      return signalPassFailure(); 
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::GlobalOptimization