// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- CPUEncodingExternalModels.cpp --------------------------------------===//
//
// This file implements the following interfaces for CPU backends and the VMVX
// backend:
//
// - IREE::Encoding::LayoutResolverAttr
// - IREE::Encoding::SerializableAttr
// - IREE::Encoding::LayoutMaterializerAttr
// - IREE::Codegen::PackedLayoutMaterializerAttr
// - VerifiableTensorEncoding
//
// In these backends, we transpose narrow-N into narrow-M
// for a combination of reasons:
//
//   1. As linalg.matmul materializes into linalg.mmt4d, which has a transposed
//      RHS and therefore LHS<->RHS symmetry, transposeNarrowN is easy to
//      implement at that level.
//   2. We use ukernels, and this allows writing 2x fewer narrow ukernels.
//   3. Heuristics for cache-friendly dispatch tiling can get complex on CPU,
//      so it is nice that they have fewer narrow cases to consider.
//
// The only current exception to this is Arm SVE. It currently adheres to the
// canonical form of scalable vectorisation and keeps the N dimension to be
// scalable.
// This transposition is made easier by (and was all along part of the idea in)
// the RHS-transposition in mmt4d (the t in mmt4d), as generally with matrix
// multiplication
//
//   B * Transpose(A) == Transpose( A * Transpose(B) )
//
// so in mmt4d terms
//
//   mmt4d(B, A) == Transpose(mmt4d(A, B))
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/ExternalInterfaces/CPUEncodingExternalModels.h"

#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "iree-codegen-materialize-encoding"

namespace mlir::iree_compiler::IREE::CPU {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Codegen::TileMxNxK;

namespace {

//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

static FailureOr<IREE::Codegen::ScalableTileFlags>
getScalableTileFlags(linalg::ContractionDimensions cDims,
                     IREE::Encoding::EncodingAttr encoding,
                     DictionaryAttr config) {
  // TODO(egebeysel): I think this isScalable*Enabled flag should be temporary
  // and the temporary SME flag should probably come next to it.
  if (!isAArch64(config) || !isScalableVectorizationEnabled()) {
    LDBG() << "Pre-conditions to enable scalable tiling are not met!";
    return failure();
  }

  std::optional<unsigned> mDim =
      cDims.m.empty() ? std::nullopt
                      : encoding.mapDimToOperandIndex(cDims.m[0]);
  std::optional<unsigned> nDim =
      cDims.n.empty() ? std::nullopt
                      : encoding.mapDimToOperandIndex(cDims.n[0]);
  std::optional<unsigned> kDim = encoding.mapDimToOperandIndex(cDims.k[0]);
  IREE::Codegen::ScalableTileFlags scalableTiles;
  // TODO(egebeysel): Add logic for SME.
  if (mDim.has_value()) {
    if (hasFeature(config, "+sme")) {
      LDBG() << "SME with data-tiling is not supported yet!";
      return failure();
    }
    scalableTiles.push_back(false);
  }
  if (nDim.has_value()) {
    scalableTiles.push_back(hasFeature(config, "+sve") ||
                            hasFeature(config, "+sve2"));
  }
  if (kDim.has_value()) {
    scalableTiles.push_back(false);
  }
  return scalableTiles;
}

static void transposeInPlace(MaterializeEncodingInfo &info) {
  // Vector cases: nothing to do.
  if (info.innerTileSizes.size() < 2) {
    return;
  }
  // Not a vector case, so all three arrays in `info` have size at least 2,
  // outerDimsPerm may have size 3 if there is a batch dimension, but in all
  // cases, the last 2 entries of each array are M and N, not batch.
  auto transpose = [](auto &a) { std::swap(a[a.size() - 2], a[a.size() - 1]); };
  transpose(info.innerDimsPos);
  transpose(info.innerTileSizes);
  transpose(info.outerDimsPerm);
  if (info.scalableTiles) {
    transpose(info.scalableTiles.value());
  }
}

static RankedTensorType
getExpandedType(RankedTensorType type, bool isBatched, bool isTransposed,
                SmallVectorImpl<ReassociationIndices> &ri) {
  if (!isBatched) {
    ri.assign({{0, 1}, {2, 3}});
    if (!isTransposed) {
      return RankedTensorType::get(
          {1, type.getDimSize(0), 1, type.getDimSize(1)},
          type.getElementType());
    }
    return RankedTensorType::get({type.getDimSize(0), 1, type.getDimSize(1), 1},
                                 type.getElementType());
  }

  ri.assign({{0}, {1, 2}, {3, 4}});
  if (!isTransposed) {
    return RankedTensorType::get(
        {type.getDimSize(0), 1, type.getDimSize(1), 1, type.getDimSize(2)},
        type.getElementType());
  }
  return RankedTensorType::get(
      {type.getDimSize(0), type.getDimSize(1), 1, type.getDimSize(2), 1},
      type.getElementType());
}

/// Given an input Value and a desired output element type, create and return
/// an element-wise linalg::GenericOp that extends the input Value to the
/// output element type. Returns `input` if casting is not needed.
static Value createElementWiseExtUIOp(OpBuilder &builder, Value input,
                                      Location loc, Type outElemType) {
  auto inputType = cast<RankedTensorType>(input.getType());
  if (inputType.getElementType() == outElemType) {
    return input;
  }
  SmallVector<AffineMap> maps(
      2, builder.getMultiDimIdentityMap(inputType.getRank()));
  SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(),
                                                 utils::IteratorType::parallel);
  auto castedType = inputType.clone(outElemType);
  SmallVector<OpFoldResult> inputMixedSizes =
      tensor::getMixedSizes(builder, loc, input);
  Value init =
      tensor::EmptyOp::create(builder, loc, inputMixedSizes, outElemType);
  return linalg::GenericOp::create(
             builder, loc, castedType, input, init, maps, iteratorTypes,
             [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
               Value castRes =
                   arith::ExtUIOp::create(b, nestedLoc, outElemType, args[0])
                       ->getResult(0);
               linalg::YieldOp::create(b, nestedLoc, castRes);
             })
      .getResult(0);
}

/// If needed, expand and the input Value, and return the resulting input with
/// the canonical mmt4d input shape. If the input element type is unsigned,
/// create a producer Linalg::GenericOp on the input that unsigned extends the
/// input to the output element type. This extension is required to keep the
/// unsignedness information on the input for ukernels. If `transpose` is true,
/// the `linalgOp`'s indexing maps are transposed.
static Value getMmt4dOperand(Value value, linalg::LinalgOp linalgOp,
                             bool transpose, OpBuilder &builder,
                             SmallVectorImpl<ReassociationIndices> &ri,
                             ArrayRef<Type> elemTypes, int operandIdx) {
  assert(linalgOp.getNumDpsInputs() == 2);
  assert(linalgOp.getNumDpsInits() == 1);
  auto cDims = linalg::inferContractionDims(linalgOp);
  Location loc = linalgOp->getLoc();
  Value expandedValue = value;
  // If vecmat with non-rhs operandIdx or matvec with non-lhs operandIdx, the
  // operand is a vector and must be extended
  if ((cDims->m.empty() && operandIdx != 1) ||
      (cDims->n.empty() && operandIdx != 0)) {
    auto type = cast<RankedTensorType>(value.getType());
    RankedTensorType newType = getExpandedType(
        type, /*isBatched=*/!cDims->batch.empty(),
        /*isTransposed=*/operandIdx == 2 && (transpose ^ cDims->n.empty()), ri);
    expandedValue =
        tensor::ExpandShapeOp::create(builder, loc, newType, value, ri);
  }
  if (elemTypes[operandIdx].isUnsignedInteger()) {
    return createElementWiseExtUIOp(builder, expandedValue, loc,
                                    elemTypes.back());
  }
  return expandedValue;
}

/// Returns the best TileMxNxK from `enumeratedTiles` pool. If the
/// `hostDefinedUpperBound` is not empty, the chosen tile sizes can not be
/// greater than the values.
TileMxNxK chooseMatmulTile(ArrayRef<TileMxNxK> enumeratedTiles,
                           IREE::Encoding::MatmulNarrowDim narrowDim) {
  // Handle narrow-N by transposing to reduce to narrow-M. Note: the
  // enumeratedTiles currently only enumerate narrow-M cases.
  if (narrowDim.isN()) {
    narrowDim.dim = IREE::Encoding::MatmulNarrowDim::Dim::M;
    TileMxNxK tile = chooseMatmulTile(enumeratedTiles, narrowDim);
    std::swap(tile.M, tile.N);
    return tile;
  }
  // Handle kDynamic: currently this is only used with VMVX, where there is only
  // one enumerated tile and it has all three M/N/K dimensions dynamic, so for
  // now we only support that. Generalize that as needed when more dynamic tile
  // sizes are used outside of VMVX, e.g. perhaps some day with Arm SVE. Decide
  // how to incorporate the handling of kDynamic in the cost-model evaluation
  // below to decide when to prefer a dynamic vs a static tile shape.
  for (auto tile : enumeratedTiles) {
    if (ShapedType::isDynamic(tile.M) || ShapedType::isDynamic(tile.N) ||
        ShapedType::isDynamic(tile.K)) {
      assert(enumeratedTiles.size() == 1);
      assert(ShapedType::isDynamic(tile.M) && ShapedType::isDynamic(tile.N) &&
             ShapedType::isDynamic(tile.K));
      return tile;
    }
  }
  // We're going to "rate" the enumerated tiles.
  struct RatedTileMxNxK : TileMxNxK {
    RatedTileMxNxK() {}
    RatedTileMxNxK(TileMxNxK tile) : TileMxNxK(tile) {}
    // Penalize tiles that are wider in the M dimension than matmulNarrowM.
    int64_t paddingPenalty = 0;
    // Favor larger tiles, as long as they still minimize paddingPenalty.
    int64_t productMxNxK = 0;
  };
  SmallVector<RatedTileMxNxK> ratedTiles;
  ratedTiles.reserve(enumeratedTiles.size());
  int64_t bestPaddingPenalty = INT64_MAX;
  for (auto tile : enumeratedTiles) {
    RatedTileMxNxK ratedTile(tile);
    ratedTile.paddingPenalty = 0;
    // If we are choosing a tile for a narrow-M case, we want to minimize
    // padding along the M dimension.
    // The PowerOf2Ceil is so that we are OK with padding up to the next
    // power of two, we just try to avoid padding beyond that. For example,
    // if matmulNarrowM==7 and we have enumerated tiles with M=8,4,2,1, we
    // are OK with the tile that has M==8 even though it requires some padding.
    // Otherwise, we would be penalizing the tiles with M==8,4,2 and we would
    // end up selecting the vecmat tile (M==1) for that case!
    if (narrowDim) {
      ratedTile.paddingPenalty =
          std::max<int64_t>(tile.M - llvm::PowerOf2Ceil(narrowDim.size), 0);
    }
    ratedTile.productMxNxK = tile.M * tile.N * tile.K;
    ratedTiles.push_back(ratedTile);
    LDBG() << "candidate: "
           << llvm::interleaved(ArrayRef{tile.M, tile.N, tile.K})
           << " penalty:" << ratedTile.paddingPenalty;
    bestPaddingPenalty = std::min(bestPaddingPenalty, ratedTile.paddingPenalty);
  }
  RatedTileMxNxK bestRatedTile;
  for (auto ratedTile : ratedTiles) {
    // Choose only among tiles that minimize paddingPenalty. Among those,
    // maximize productMxNxK.
    if (ratedTile.paddingPenalty == bestPaddingPenalty &&
        bestRatedTile.productMxNxK < ratedTile.productMxNxK) {
      bestRatedTile = ratedTile;
    }
  }
  // Sanity check. This assert can only fail if there's a programming mistake
  // locally here.
  assert(bestRatedTile.paddingPenalty == bestPaddingPenalty);
  LDBG() << "bestRatedTile: "
         << llvm::interleaved(
                ArrayRef{bestRatedTile.M, bestRatedTile.N, bestRatedTile.K})
         << " penalty:" << bestRatedTile.paddingPenalty;
  return bestRatedTile;
}

Operation *lowerContractionOpWithEncoding(
    OpBuilder &builder, linalg::LinalgOp linalgOp, ValueRange operands,
    IREE::Encoding::LayoutMaterializerAttr layoutAttr) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return nullptr;
  }

  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();

  auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
  auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
  auto resultType = cast<RankedTensorType>(outputs[0].getType());
  auto lhsEncoding = IREE::Encoding::getEncodingAttr(lhsType);
  auto rhsEncoding = IREE::Encoding::getEncodingAttr(rhsType);
  auto resultEncoding = IREE::Encoding::getEncodingAttr(resultType);
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return nullptr;
  }

  if (lhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_LHS ||
      rhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RHS ||
      resultEncoding.getOperandIndex().getValue() !=
          IREE::Encoding::MATMUL_RESULT) {
    return nullptr;
  }

  MaterializeEncodingInfo encodingInfo = {};
  if (auto packedLayoutAttr =
          dyn_cast<IREE::Codegen::PackedLayoutMaterializerAttr>(layoutAttr)) {
    encodingInfo = packedLayoutAttr.getEncodingInfo(
        cast<RankedTensorType>(linalgOp->getResultTypes()[0]));
  }

  if (isIdentityLayout(encodingInfo)) {
    return dropEncodingAndCloneOp(builder, linalgOp,
                                  operands.take_front(inputs.size()),
                                  operands.drop_front(inputs.size()));
  }

  auto cDims = IREE::Encoding::getEncodingContractionDims(lhsEncoding);
  bool transpose = isNarrowNResult(resultEncoding);
  // Do not transpose in case we have scalable tiles.
  transpose &= llvm::none_of(
      encodingInfo.scalableTiles.value_or(IREE::Codegen::ScalableTileFlags{}),
      [](bool flag) { return flag; });
  // Batch matmuls: never transpose. Transposing swaps LHS/RHS operands but
  // not the Result, which breaks batch_mmt4d dimension verification when M!=N.
  if (succeeded(cDims) && !cDims->batch.empty()) {
    transpose = false;
  }
  SmallVector<Type> elemTypes = lhsEncoding.getElementTypesArray();
  SmallVector<ReassociationIndices> ri;
  Value newLhs = getMmt4dOperand(operands[0], linalgOp, transpose, builder, ri,
                                 elemTypes, /*operandIdx=*/0);
  Value newRhs = getMmt4dOperand(operands[1], linalgOp, transpose, builder, ri,
                                 elemTypes, /*operandIdx=*/1);
  Value newResult = getMmt4dOperand(operands[2], linalgOp, transpose, builder,
                                    ri, elemTypes, /*operandIdx=*/2);
  if (transpose) {
    std::swap(newLhs, newRhs);
  }
  Type newResultType = newResult.getType();
  Operation *result;
  if (failed(cDims) || cDims->batch.empty()) {
    result = linalg::Mmt4DOp::create(builder, linalgOp.getLoc(), newResultType,
                                     ValueRange{newLhs, newRhs},
                                     ValueRange{newResult});
  } else {
    result = linalg::BatchMmt4DOp::create(
        builder, linalgOp.getLoc(), newResultType, ValueRange{newLhs, newRhs},
        ValueRange{newResult});
  }
  // Propagate OPU transposed-output annotation from the original matmul.
  if (linalgOp->hasAttr("iree.opu_transposed_output")) {
    result->setAttr("iree.opu_transposed_output",
                    builder.getUnitAttr());
  }
  if (!ri.empty()) {
    result = tensor::CollapseShapeOp::create(builder, linalgOp->getLoc(),
                                             operands[2].getType(),
                                             result->getResult(0), ri);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Interface methods implementation for iree_cpu.cpu_encoding_resolver.
//===----------------------------------------------------------------------===//

// Enumerate tile sizes to choose from on riscv32.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTileRiscv32(DictionaryAttr config) {
  if (hasUkernel(config)) {
    return {
        TileMxNxK{8, 8, 4}, // Some reasonable tile shape.
        TileMxNxK{4, 8, 4}, // Truncation of the above.
        TileMxNxK{2, 8, 4}, // Truncation of the above.
        TileMxNxK{1, 8, 4}, // Truncation of the above.
    };
  }
  // Fallback - no architecture-optimized tile size for this case.
  return {};
}
// RISC-V has vector register length extensions: zvl128b, zvl256b etc.
// If these extension are specified in target cpu feature,
// they can be used to determine VLEN. This function assumes that
// 'v' feature is present
size_t getRISCVVVlenFromCPUFeatures(DictionaryAttr config) {
  // If +zvl* feature is not explicitly specified,
  // fallback to +zvl128b, as spec specifies minimum VLEN
  // of 128b for the V extension: https://rb.gy/p8rbzv
  size_t vlen;
  if (hasFeature(config, "+zvl65536b")) {
    vlen = 65536;
  } else if (hasFeature(config, "+zvl32768b")) {
    vlen = 32768;
  } else if (hasFeature(config, "+zvl16384b")) {
    vlen = 16384;
  } else if (hasFeature(config, "+zvl8192b")) {
    vlen = 8192;
  } else if (hasFeature(config, "+zvl4096b")) {
    vlen = 4096;
  } else if (hasFeature(config, "+zvl2048b")) {
    vlen = 2048;
  } else if (hasFeature(config, "+zvl1024b")) {
    vlen = 1024;
  } else if (hasFeature(config, "+zvl512b")) {
    vlen = 512;
  } else if (hasFeature(config, "+zvl256b")) {
    vlen = 256;
  } else {
    vlen = 128;
  }
  return vlen;
}
// Enumerate tile sizes to choose from on riscv64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateMatmulTileRiscv64(TypeRange elementTypes, DictionaryAttr config) {
  // Data-Tiling is only implemented for the V extension
  if (!hasFeature(config, "+v")) {
    return {};
  }
  size_t vlen = getRISCVVVlenFromCPUFeatures(config);
  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];

  // VLEN-aware Tile size selection.
  // One concern that needs to be addressed here is that
  // for larger VLENs tile sizes would be very large
  // leading to a very high padding overhead.
  if (lhs.isF32() && rhs.isF32() && out.isF32()) {
    int N0 = vlen / 8;
    return {
        TileMxNxK{7, N0, 1}, // Aim to use vfmacc, 100% register utilization.
        TileMxNxK{4, N0, 1}, // Truncation of the above.
        TileMxNxK{2, N0, 1}, // Truncation of the above.
        TileMxNxK{1, N0, 1}, // Truncation of the above.
    };
  }
  if (lhs.isF16() && rhs.isF16()) {
    int N0 = vlen / 8;
    if (hasFeature(config, "+zvfh")) {
      return {
          TileMxNxK{7, N0, 1},
          TileMxNxK{4, N0, 1}, // Truncation of the above.
          TileMxNxK{2, N0, 1}, // Truncation of the above.
          TileMxNxK{1, N0, 1}, // Truncation of the above.
      };
    }
    if (hasFeature(config, "+zvfhmin")) {
      return {
          TileMxNxK{6, N0, 1},
          TileMxNxK{4, N0, 1}, // Truncation of the above.
          TileMxNxK{2, N0, 1}, // Truncation of the above.
          TileMxNxK{1, N0, 1}, // Truncation of the above.
      };
    }
  }

  auto isFp8Type = [](Type type) -> bool {
    auto floatType = dyn_cast<FloatType>(type);
    return floatType && floatType.getWidth() == 8;
  };

  // FP8 paths.
  if (isFp8Type(lhs) && isFp8Type(rhs) &&
      isa<FloatType>(out) && out.getIntOrFloatBitWidth() >= 16) {
    if (hasFeature(config, "+xopu")) {
      // OPU FP8: OPFMACC uses same 16x16 matrix engine as int8 VOPACC.
      // K0=16: each vlse8 has stride=16 (1 cache line), good locality.
      // The compiler tiles K into K1*K0 so total K work is preserved.
      int N0 = std::min<int>(16, vlen / 8);
      return {
          TileMxNxK{N0, N0, 16},
          TileMxNxK{N0 / 2, N0, 16},
          TileMxNxK{N0 / 4, N0, 16},
          TileMxNxK{1, N0, 16},
      };
    }
    if (hasFeature(config, "+xsmtvdot")) {
      return {
          TileMxNxK{4, 4, 8},
          TileMxNxK{2, 4, 8},
          TileMxNxK{1, 4, 8},
      };
    }
  }

  // Integer 8 path.
  if (lhs.isSignlessInteger(8) && rhs.isSignlessInteger(8) &&
      out.isSignlessInteger(32)) {

    // Saturn OPU: 16×16 hardware tiles. Tested 2026-05-28 (Lane B):
    // larger 32×32 / 32×16 encoding tiles trip an MLIR vector-size
    // cap (8192-byte vector ops), even though the OPU ukernel itself
    // handles 32×32 via 2×2 sub-tiling at runtime. 16×32 (wider N
    // only) also fails on im2col paths under the OPU encoding
    // resolver. Keeping the conservative 16×16 menu.
    if (hasFeature(config, "+xopu")) {
      int N0 = std::min<int>(16, vlen / 8);  // e8, m1 lanes = 16 on V128
      return {
          TileMxNxK{N0, N0, 1},           // 16x16: OPU hardware tile
          TileMxNxK{N0 / 2, N0, 1},       // 8x16: narrow-M
          TileMxNxK{N0 / 4, N0, 1},       // 4x16: narrower-M
          TileMxNxK{1, N0, 1},            // 1x16: vecmat
      };
    }

    if (hasFeature(config, "+xsmtvdot")) {
      // SpacemiT vmadot: fixed 4x4x8 blocks.
      return {
          TileMxNxK{4, 4, 8},
          TileMxNxK{2, 4, 8},
          TileMxNxK{1, 4, 8},
      };
    }

    // Standard RVV path:
    // N0 calculation:
    // vlen (bits) / 32 (bits per element) = elements per register (LMUL1)
    // We target LMUL2 here (N0=16 on VLEN=256) to reduce accumulator pressure
    // and allow deeper M unrolling.
    int elementsPerReg = vlen / 32;
    int N0 = elementsPerReg * 2; // Target LMUL=2 (e.g., 16 elements)

    return {
        // Prefer wider M when possible; chooseMatmulTile will pick the best fit.
        TileMxNxK{8, N0, 1},
        TileMxNxK{7, N0, 1},
        TileMxNxK{4, N0, 1},
        TileMxNxK{2, N0, 1},
        TileMxNxK{1, N0, 1},
    };
  }

  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

// Enumerate tile sizes to choose from on arm64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK> enumerateMatmulTileArm64(TypeRange elementTypes,
                                                       DictionaryAttr config) {
  // For SVE and scalable vectors, this methods selects base sizes that match
  // the NEON fixed-width sizes.
  // TODO: Add SME inner tile sizes and corresponding tests.
  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];

  if (out.isF32() || out.isF16() || out.isBF16()) {
    if (lhs.isBF16() && rhs.isBF16() && (out.isBF16() || out.isF32()) &&
        hasFeature(config, "+bf16")) {
      return {
          TileMxNxK{8, 8, 4}, // Aim to use BFMMLA.
          TileMxNxK{4, 8, 4}, // Truncation of the above.
          TileMxNxK{2, 8, 4}, // Truncation of the above.
          TileMxNxK{1, 8, 4}, // Truncation of the above.
      };
    }
    if (isa<FloatType>(lhs) && isa<FloatType>(rhs)) {
      // Note: 16-bit floating point types currently use the same tile size as
      // f32. This makes sense when either (1) the accumulator is f32, or (2)
      // the arithmetic will have to expand f16 to f32 in registers. We may
      // reconsider when taking advantage of native f16/bf16 arithmetic when the
      // accumulator itself is f16/bf16, as we could typically have a 2x wider
      // tile in that case. However, on current CPUs, the existing tiles seem
      // wide enough already to approach peak performance.
      return {
          TileMxNxK{8, 8, 1}, // Aim to use FMLA or FMLAL.
          TileMxNxK{4, 8, 1}, // Truncation of the above.
          TileMxNxK{2, 8, 1}, // Truncation of the above.
          TileMxNxK{1, 8, 1}, // Truncation of the above.
      };
    }
  }
  if (lhs.isSignlessInteger(8) && rhs.isSignlessInteger(8) &&
      out.isSignlessInteger(32)) {
    if (hasFeature(config, "+i8mm")) {
      return {
          TileMxNxK{8, 8, 8}, // Aim to use SMMLA.
          TileMxNxK{4, 8, 8}, // Truncation of the above.
          TileMxNxK{2, 8, 8}, // Truncation of the above.
          TileMxNxK{1, 8, 8}, // Truncation of the above.
      };
    }
    if (hasFeature(config, "+dotprod")) {
      return {
          TileMxNxK{8, 8, 4}, // Aim to use SDOT.
          TileMxNxK{4, 8, 4}, // Truncation of the above.
          TileMxNxK{2, 8, 4}, // Truncation of the above.
          TileMxNxK{1, 8, 4}, // Truncation of the above.
      };
    }
  }
  if (lhs.isSignlessInteger(8) && rhs.isSignlessInteger(4) &&
      out.isSignlessInteger(32)) {
    if (hasFeature(config, "+i8mm")) {
      return {
          TileMxNxK{4, 8, 16},
          TileMxNxK{2, 8, 16},
          TileMxNxK{1, 8, 16},
      };
    }
    if (hasFeature(config, "+dotprod")) {
      return {
          TileMxNxK{8, 8, 8},
          TileMxNxK{4, 8, 8},
          TileMxNxK{2, 8, 8},
          TileMxNxK{1, 8, 8},
      };
    }
    return {
        TileMxNxK{4, 16, 2},
        TileMxNxK{2, 16, 2},
        TileMxNxK{1, 16, 2},
    };
  }
  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

// Enumerate tile sizes to choose from on x86-64.
// For narrow-{M,N} cases, this only enumerates on narrow M. The narrow-N cases
// are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK> enumerateMatmulTileX86_64(TypeRange elementTypes,
                                                        DictionaryAttr config) {
  assert(elementTypes.size() == 3);
  Type lhs = elementTypes[0];
  Type rhs = elementTypes[1];
  Type out = elementTypes[2];

  if (out.isF32() || out.isF16() || out.isBF16()) {
    if (lhs.isBF16() && rhs.isBF16() && (out.isBF16() || out.isF32())) {
      if (hasFeature(config, "+avx512bf16")) {
        return {
            TileMxNxK{16, 16, 2}, // Aim to use VDPBF16PS (zmm).
            TileMxNxK{8, 16, 2},  // Truncation of the above.
            TileMxNxK{4, 16, 2},  // Truncation of the above.
            TileMxNxK{2, 16, 2},  // Truncation of the above.
            TileMxNxK{1, 16, 2},  // Truncation of the above.
        };
      }
    }
    if (isa<FloatType>(lhs) && isa<FloatType>(rhs)) {
      // Note: 16-bit floating point types currently use the same tile size as
      // f32. This makes sense when either (1) the accumulator is f32, or (2)
      // the arithmetic will have to expand f16 to f32 in registers. We may
      // reconsider when taking advantage of native f16/bf16 arithmetic when the
      // accumulator itself is f16/bf16.
      if (hasFeature(config, "+avx512f")) {
        return {
            TileMxNxK{16, 16, 1}, // Aim to use VFMADD* (zmm).
            TileMxNxK{8, 16, 1},  // Truncation of the above.
            TileMxNxK{4, 16, 1},  // Truncation of the above.
            TileMxNxK{2, 16, 1},  // Truncation of the above.
            TileMxNxK{1, 16, 1},  // Truncation of the above.
        };
      }
      if (hasFeature(config, "+avx")) {
        // Note: for good performance, most +avx users will also want to add
        // +fma, but that's a local instruction selection detail and the tile
        // layout is unaffected, as there are enough registers even with the
        // need for intermediate product registers when +fma is not used.
        return {
            TileMxNxK{8, 8, 1}, // Aim to use VFMADD* (ymm).
            TileMxNxK{4, 8, 1}, // Truncation of the above.
            TileMxNxK{2, 8, 1}, // Truncation of the above.
            TileMxNxK{1, 8, 1}, // Truncation of the above.
        };
      }
      // SSE fallback.
      return {
          TileMxNxK{8, 4, 1}, // Aim to use MULPS/ADDPS (xmm).
          TileMxNxK{4, 4, 1}, // Truncation of the above.
          TileMxNxK{2, 4, 1}, // Truncation of the above.
          TileMxNxK{1, 4, 1}, // Truncation of the above.
      };
    }
  }

  if (out.isSignlessInteger(32) &&
      ((lhs.isSignlessInteger(8) && rhs.isSignlessInteger(8)) ||
       (lhs.isSignlessInteger(16) && rhs.isSignlessInteger(16)))) {
    if (hasFeature(config, "+avx512vnni")) {
      // This is the same tile size as with VPMADDWD as the only difference
      // is that VPDPWSSD accumulates. VPDPBUSD would call for {16, 16, 4} but
      // we can't easily use it because of its unsigned*signed semantics.
      return {
          TileMxNxK{16, 16, 2}, // Aim to use VPDPWSSD (zmm).
          TileMxNxK{8, 16, 2},  // Truncation of the above.
          TileMxNxK{4, 16, 2},  // Truncation of the above.
          TileMxNxK{2, 16, 2},  // Truncation of the above.
          TileMxNxK{1, 16, 2},  // Truncation of the above.
      };
    }
    if (hasFeature(config, "+avx512bw")) {
      return {
          TileMxNxK{16, 16, 2}, // Aim to use VPMADDWD (zmm).
          TileMxNxK{8, 16, 2},  // Truncation of the above.
          TileMxNxK{4, 16, 2},  // Truncation of the above.
          TileMxNxK{2, 16, 2},  // Truncation of the above.
          TileMxNxK{1, 16, 2},  // Truncation of the above.
      };
    }
    if (hasFeature(config, "+avx2")) {
      return {
          TileMxNxK{8, 8, 2}, // Aim to use VPMADDWD (ymm).
          TileMxNxK{4, 8, 2}, // Truncation of the above.
          TileMxNxK{2, 8, 2}, // Truncation of the above.
          TileMxNxK{1, 8, 2}, // Truncation of the above.
      };
    }
    // SSE fallback.
    return {
        TileMxNxK{8, 4, 2}, // Aim to use PMADDWD (xmm).
        TileMxNxK{4, 4, 2}, // Truncation of the above.
        TileMxNxK{2, 4, 2}, // Truncation of the above.
        TileMxNxK{1, 4, 2}, // Truncation of the above.
    };
  }

  if (out.isSignlessInteger(32) && lhs.isSignlessInteger(16) &&
      rhs.isUnsignedInteger(4)) {
    // Experimental s16u4s32 case. Focusing only on the vecmat case for now.
    if (hasFeature(config, "+avx512vnni")) {
      return {
          TileMxNxK{1, 32, 8}, // Aim to use VPDPBUSD (zmm).
      };
    }
  }

  // Fallback - no architecture-optimized tile size for this case.
  return {};
}

static SmallVector<TileMxNxK>
enumerateCPUMatmulTiles(IREE::Encoding::EncodingAttr encoding,
                        DictionaryAttr config) {
  // Enumerate available tile shapes for the given encoding and config.
  SmallVector<Type> elementTypes = encoding.getElementTypesArray();
  if (isAArch64(config)) {
    return enumerateMatmulTileArm64(elementTypes, config);
  }
  if (isX86_64(config)) {
    return enumerateMatmulTileX86_64(elementTypes, config);
  }
  if (isRISCV32(config)) {
    return enumerateMatmulTileRiscv32(config);
  }
  if (isRISCV64(config)) {
    return enumerateMatmulTileRiscv64(elementTypes, config);
  }
  return {};
}

struct CPUEncodingPackedLayoutMaterializerAttr
    : PackedLayoutMaterializerAttrExternalModelBase<
          CPUEncodingPackedLayoutMaterializerAttr, CPUEncodingResolverAttr> {

  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<CPUEncodingResolverAttr>(attr).getConfiguration();
  }

  MaterializeEncodingInfo getEncodingInfoImpl(Attribute attr,
                                              RankedTensorType type) const {
    auto layoutAttr = cast<CPUEncodingResolverAttr>(attr);

    auto encoding =
        dyn_cast_if_present<IREE::Encoding::EncodingAttr>(type.getEncoding());

    MaterializeEncodingInfo info;
    if (!encoding) {
      return info;
    }

    // We only know about contractions with {Batch, M, N, K} <= 1 at the moment.
    auto cDims = getEncodingContractionDims(encoding);
    if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
        cDims->n.size() > 1 || cDims->k.size() > 1) {
      return info;
    }

    SmallVector<TileMxNxK> enumeratedTileMxNxK =
        enumerateCPUMatmulTiles(encoding, layoutAttr.getConfiguration());
    if (enumeratedTileMxNxK.empty()) {
      return info;
    }
    auto narrowDim = IREE::Encoding::getPo2MatmulNarrowDim(encoding);
    // Choose a final matmul TileMxNxK from the above-enumerated tile shapes,
    // taking narrow dimensions into account.
    TileMxNxK chosenTileMxNxK =
        chooseMatmulTile(enumeratedTileMxNxK, narrowDim);
    FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
        getEncodingInfoForMatmul(encoding, chosenTileMxNxK);
    if (failed(maybeEncodingInfo)) {
      return info;
    }
    info = std::move(maybeEncodingInfo.value());

    FailureOr<IREE::Codegen::ScalableTileFlags> scalableFlags =
        getScalableTileFlags(*cDims, encoding, layoutAttr.getConfiguration());
    if (succeeded(scalableFlags)) {
      info.scalableTiles = std::move(scalableFlags);
    }
    if (IREE::Encoding::isNarrowNResult(encoding) &&
        llvm::none_of(info.scalableTiles.value_or(Codegen::ScalableTileFlags{}),
                      [](bool flag) { return flag; })) {
      transposeInPlace(info);
    }
    return info;
  }
};

struct CPUEncodingResolverMaterializerAttr final
    : EncodingLayoutMaterializerAttrExternalModelBase<
          CPUEncodingResolverMaterializerAttr, CPUEncodingResolverAttr> {

  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto layoutAttr = cast<CPUEncodingResolverAttr>(attr);
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }
    if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      return lowerFillOpWithResolvedLayouts(b, fillOp, convertedResTypes,
                                            convertedOperands);
    }
    // Scaled contraction (MX matmul) is not yet supported on CPU, so we drop
    // the encoding and clone the op as-is.
    if (IREE::LinalgExt::isaScaledContractionOpInterface(linalgOp)) {
      int64_t numInputs = linalgOp.getNumDpsInputs();
      return dropEncodingAndCloneOp(b, linalgOp,
                                    convertedOperands.take_front(numInputs),
                                    convertedOperands.drop_front(numInputs));
    }
    if (linalg::isaContractionOpInterface(linalgOp)) {
      return lowerContractionOpWithEncoding(
          b, linalgOp, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      return lowerGenericOpWithResolvedLayouts(
          b, genericOp, convertedResTypes, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(attr));
    }
    return nullptr;
  }
};

struct CPULayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<
          CPULayoutResolverAttr, CPUEncodingResolverAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    SmallVector<NamedAttribute> configItems;
    if (std::optional<StringRef> cpuFeatures = getConfigCpuFeatures(config)) {
      addConfigCpuFeatures(ctx, cpuFeatures.value(), configItems);
    }
    if (std::optional<StringRef> targetTriple = getConfigTargetTriple(config)) {
      addConfigTargetTriple(ctx, targetTriple.value(), configItems);
    }
    storeNamedAttrIfPresent(configItems, config, "ukernels");
    return CPUEncodingResolverAttr::get(ctx,
                                        DictionaryAttr::get(ctx, configItems));
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    return CPUEncodingResolverAttr::get(ctx, getPackedLayoutImpl(attr, type));
  }
};

struct CPUSerializableAttr final
    : IREE::Encoding::SerializableAttr::ExternalModel<CPUSerializableAttr,
                                                      CPUEncodingResolverAttr> {
  bool isSerialized(Attribute attr) const {
    auto configuration = cast<CPUEncodingResolverAttr>(attr).getConfiguration();
    return configuration && configuration.contains(kEncodingInfoAttrName);
  }

  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    return calculatePackedStorageSizeInBytesImpl(attr, loc, builder, type,
                                                 dynamicDims);
  }
};

struct CPUEncodingResolverVerifier
    : mlir::VerifiableTensorEncoding::ExternalModel<CPUEncodingResolverVerifier,
                                                    CPUEncodingResolverAttr> {

  LogicalResult
  verifyEncoding(Attribute attr, ArrayRef<int64_t> shape, Type elementType,
                 function_ref<InFlightDiagnostic()> emitError) const {
    auto packedLayoutMaterializerAttr =
        cast<Codegen::PackedLayoutMaterializerAttr>(attr);
    return packedLayoutMaterializerAttr.verifyPackedLayoutWithType(
        shape, elementType, emitError);
  }
};

//===----------------------------------------------------------------------===//
// Interface methods implementation for iree_cpu.vmvx_encoding_resolver.
//===----------------------------------------------------------------------===//

// Enumerate tile sizes to choose from when no specific architecture is
// targeted. For narrow-{M,N} cases, this only enumerates on narrow M. The
// narrow-N cases are handled by transposition in chooseMatmulTile.
static SmallVector<TileMxNxK>
enumerateVMVXMatmulTiles(linalg::ContractionDimensions cDims,
                         IREE::Encoding::EncodingAttr encoding,
                         DictionaryAttr config) {
  bool hasUkernelSupport = hasUkernel(config);

  // TODO(hanchung): The ukernel path does not support 3d
  // codegen.query_tile_sizes op, so we disable dynamic tile shapes for
  // batch_matmul. Also, they are not set up for narrow M/N matmul, so it is
  // disabled when it is the case.
  if (!cDims.batch.empty() || getPo2MatmulNarrowDim(encoding)) {
    hasUkernelSupport = false;
  }
  if (hasUkernelSupport) {
    // VMVX+ukernel uses dynamic tile shapes.
    return {TileMxNxK{ShapedType::kDynamic, ShapedType::kDynamic,
                      ShapedType::kDynamic}};
  }

  return {
      TileMxNxK{8, 8, 4}, // Some vaguely reasonable tile shape.
      TileMxNxK{4, 8, 4}, // Truncation of the above.
      TileMxNxK{2, 8, 4}, // Truncation of the above.
      TileMxNxK{1, 8, 4}, // Truncation of the above.
  };
}

struct VMVXEncodingPackedLayoutMaterializerAttr final
    : PackedLayoutMaterializerAttrExternalModelBase<
          VMVXEncodingPackedLayoutMaterializerAttr, VMVXEncodingResolverAttr> {

  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<VMVXEncodingResolverAttr>(attr).getConfiguration();
  }

  MaterializeEncodingInfo getEncodingInfoImpl(Attribute attr,
                                              RankedTensorType type) const {
    auto layoutAttr = cast<VMVXEncodingResolverAttr>(attr);

    auto encoding =
        dyn_cast_if_present<IREE::Encoding::EncodingAttr>(type.getEncoding());

    MaterializeEncodingInfo info;
    if (!encoding) {
      return info;
    }

    // We only know about contractions with {Batch, M, N, K} <= 1 at the moment.
    auto cDims = getEncodingContractionDims(encoding);
    if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
        cDims->n.size() > 1 || cDims->k.size() > 1) {
      return info;
    }

    SmallVector<TileMxNxK> enumeratedTileMxNxK = enumerateVMVXMatmulTiles(
        cDims.value(), encoding, layoutAttr.getConfiguration());
    if (enumeratedTileMxNxK.empty()) {
      return info;
    }
    auto narrowDim = IREE::Encoding::getPo2MatmulNarrowDim(encoding);
    // Choose a final matmul TileMxNxK from the above-enumerated tile shapes,
    // taking narrow dimensions into account.
    TileMxNxK chosenTileMxNxK =
        chooseMatmulTile(enumeratedTileMxNxK, narrowDim);
    FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
        getEncodingInfoForMatmul(encoding, chosenTileMxNxK);
    if (failed(maybeEncodingInfo)) {
      return info;
    }
    info = std::move(maybeEncodingInfo.value());
    if (IREE::Encoding::isNarrowNResult(encoding)) {
      transposeInPlace(info);
    }
    return info;
  }
};

struct VMVXEncodingResolverMaterializerAttr final
    : EncodingLayoutMaterializerAttrExternalModelBase<
          VMVXEncodingResolverMaterializerAttr, VMVXEncodingResolverAttr> {

  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto layoutAttr = cast<VMVXEncodingResolverAttr>(attr);
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }
    if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      return lowerFillOpWithResolvedLayouts(b, fillOp, convertedResTypes,
                                            convertedOperands);
    }
    if (linalg::isaContractionOpInterface(linalgOp)) {
      return lowerContractionOpWithEncoding(
          b, linalgOp, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      return lowerGenericOpWithResolvedLayouts(
          b, genericOp, convertedResTypes, convertedOperands,
          cast<IREE::Encoding::LayoutMaterializerAttr>(attr));
    }
    return nullptr;
  }
};

struct VMVXLayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<
          VMVXLayoutResolverAttr, VMVXEncodingResolverAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    SmallVector<NamedAttribute> configItems;
    storeNamedAttrIfPresent(configItems, config, "ukernels");
    return VMVXEncodingResolverAttr::get(ctx,
                                         DictionaryAttr::get(ctx, configItems));
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    return VMVXEncodingResolverAttr::get(
        ctx, getPackedLayoutImpl(attr, type, /*addEncodingAttr=*/true));
  }
};

struct VMVXSerializableAttr final
    : IREE::Encoding::SerializableAttr::ExternalModel<
          VMVXSerializableAttr, VMVXEncodingResolverAttr> {
  bool isSerialized(Attribute attr) const {
    auto configuration =
        cast<VMVXEncodingResolverAttr>(attr).getConfiguration();
    return configuration && configuration.contains(kEncodingInfoAttrName);
  }

  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    return calculatePackedStorageSizeInBytesImpl(attr, loc, builder, type,
                                                 dynamicDims);
  }
};

struct VMVXEncodingResolverVerifier
    : mlir::VerifiableTensorEncoding::ExternalModel<
          VMVXEncodingResolverVerifier, VMVXEncodingResolverAttr> {
  LogicalResult
  verifyEncoding(Attribute attr, ArrayRef<int64_t> shape, Type elementType,
                 function_ref<InFlightDiagnostic()> emitError) const {
    auto packedLayoutMaterializerAttr =
        cast<Codegen::PackedLayoutMaterializerAttr>(attr);
    return packedLayoutMaterializerAttr.verifyPackedLayoutWithType(
        shape, elementType, emitError);
  }
};

// ===== Saturn OPU (+xopu) BEGIN =====
//===----------------------------------------------------------------------===//
// Interface methods implementation for iree_cpu.opu_encoding_resolver.
//===----------------------------------------------------------------------===//

/// Resilient fallback: clone the op with convertedOperands and
/// convertedResTypes, stripping any remaining encoding from the IR.
/// Used when all encoding-aware lowering paths fail (e.g., im2col shapes).
static Operation *dropAllEncodingsAndClone(OpBuilder &builder, Operation *op,
                                           ValueRange convertedOperands,
                                           TypeRange convertedResTypes) {
  // Strip encoding from result types.
  SmallVector<Type> strippedResTypes;
  for (auto type : convertedResTypes) {
    if (auto rtt = dyn_cast<RankedTensorType>(type)) {
      strippedResTypes.push_back(rtt.dropEncoding());
    } else {
      strippedResTypes.push_back(type);
    }
  }
  return mlir::clone(builder, op, strippedResTypes, convertedOperands);
}

// Helpers for OPU ukernel lowering (duplicated from CPULowerToUKernels.cpp
// because those are file-local static functions).

/// Returns `true` if an `outsOperand` value is initialized to zero.
static bool opuIsInitializedToZero(Value outsOperand) {
  auto fillOp = outsOperand.getDefiningOp<linalg::FillOp>();
  if (!fillOp) {
    return false;
  }
  Value fillVal = fillOp.getDpsInputOperand(0)->get();
  return matchPattern(fillVal, m_Zero()) ||
         matchPattern(fillVal, m_AnyZeroFloat());
}

/// Creates a UKernelGenericOp for the OPU matmul contraction.
/// Takes packed LHS/RHS inputs and produces 2D output directly.
static Operation *lowerOPUContractionToUkernel(
    OpBuilder &b, linalg::LinalgOp linalgOp, ValueRange convertedOperands) {
  Location loc = linalgOp.getLoc();

  // convertedOperands layout:
  //   [0] = packed LHS (e.g., tensor<?x?x32x1xi8>)
  //   [1] = packed RHS (e.g., tensor<?x?x32x1xi8>)
  //   [2] = 2D output  (tensor<?x?xi32>, identity encoding)
  Value packedLHS = convertedOperands[0];
  Value packedRHS = convertedOperands[1];
  Value output2D = convertedOperands[2];

  auto lhsType = cast<RankedTensorType>(packedLHS.getType());
  auto rhsType = cast<RankedTensorType>(packedRHS.getType());
  auto outType = cast<ShapedType>(output2D.getType());

  // The OPU ukernel expects 4D packed inputs [M, K, M0, K0] / [N, K, N0, K0].
  // If the encoding produced different ranks (e.g., vecmat with identity LHS,
  // or CPU-style tiles), bail out to the standard contraction path.
  if (lhsType.getRank() != 4 || rhsType.getRank() != 4) {
    return nullptr;
  }


  // Extract tile counts (dynamic) and tile sizes (static) from packed shapes.
  // LHS shape: [M, K, M0, K0]
  // RHS shape: [N, K, N0, K0]
  Value M = tensor::DimOp::create(b, loc, packedLHS, 0);
  Value K = tensor::DimOp::create(b, loc, packedLHS, 1);
  Value N = tensor::DimOp::create(b, loc, packedRHS, 0);

  auto getDimAsI32 = [](OpBuilder &b, Location loc, Value value,
                        int dim) -> Value {
    return arith::IndexCastOp::create(
        b, loc, b.getI32Type(), tensor::DimOp::create(b, loc, value, dim));
  };
  Value M0 = getDimAsI32(b, loc, packedLHS, 2);
  Value N0 = getDimAsI32(b, loc, packedRHS, 2);
  Value K0 = getDimAsI32(b, loc, packedRHS, 3);

  // Build flags.
  uint32_t flags = IREE_UK_FLAG_MMT4D_TYPE_S8S8S32;

  // Check if the accumulator is zero-filled.
  Value ukernelOut = output2D;
  if (opuIsInitializedToZero(output2D)) {
    // Output is zero-initialized; skip accumulate flag and use the fill's init.
    if (auto fillOp = output2D.getDefiningOp<linalg::FillOp>()) {
      ukernelOut = fillOp.getDpsInitOperand(0)->get();
    }
  } else {
    flags |= IREE_UK_FLAG_MMT4D_ACCUMULATE;
  }

  flags |= IREE_UK_FLAG_MMT4D_ALLOW_GENERIC_FALLBACK_TILE_FUNCTION;

  Value flagsVal =
      arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(flags));

  // Build function name and def attrs for the OPU ukernel.
  std::string fnName = "iree_uk_opu_matmul";
  SmallVector<NamedAttribute> defAttrs;
  defAttrs.emplace_back(b.getStringAttr("hal.import.fields"),
                        b.getArrayAttr({b.getStringAttr("processor_data")}));
  defAttrs.emplace_back(b.getStringAttr("hal.import.bitcode"),
                        b.getBoolAttr(true));

  // Return types: tensor result + i32 dummy (llvm-cpu void-return workaround).
  // After bufferization the tensor result disappears, leaving only the i32.
  SmallVector<Type> returnTypes{outType,
                                IntegerType::get(b.getContext(), 32)};

  auto genericOp = IREE::Codegen::UKernelGenericOp::create(
      b, loc, returnTypes, fnName, ValueRange{packedLHS, packedRHS},
      ukernelOut, ValueRange{M, N, K, M0, N0, K0, flagsVal},
      /*fn_def_attrs=*/b.getDictionaryAttr(defAttrs),
      /*num_strided_outer_dims=*/1);

  // MaterializeLinalgOp::matchAndRewrite does:
  //   rewriter.replaceOp(matmulOp, newOp->getResults())
  // The original matmul has 1 result, but UKernelGenericOp has 2 (tensor +
  // i32 dummy). We need to return a 1-result op for the replacement.
  // Use tensor.cast (identity) to forward just the tensor result.
  auto castOp = tensor::CastOp::create(b, loc, outType,
                                        genericOp.getResult(0));
  return castOp.getOperation();
}

struct OPUEncodingPackedLayoutMaterializerAttr final
    : PackedLayoutMaterializerAttrExternalModelBase<
          OPUEncodingPackedLayoutMaterializerAttr, OPUEncodingResolverAttr> {

  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<OPUEncodingResolverAttr>(attr).getConfiguration();
  }

  MaterializeEncodingInfo getEncodingInfoImpl(Attribute attr,
                                              RankedTensorType type) const {
    auto layoutAttr = cast<OPUEncodingResolverAttr>(attr);

    auto encoding =
        dyn_cast_if_present<IREE::Encoding::EncodingAttr>(type.getEncoding());

    MaterializeEncodingInfo info;
    if (!encoding) {
      return info;
    }

    auto cDims = getEncodingContractionDims(encoding);
    if (failed(cDims) || cDims->m.size() > 1 ||
        cDims->n.size() > 1 || cDims->k.size() > 1) {
      return info;
    }

    // Batch matmuls (attention QKV) and vecmat/matvec need CPU tile fallback.
    // OPU tiles only work for 2D non-batched matmuls with M,N >= 2.
    auto narrowDim = IREE::Encoding::getPo2MatmulNarrowDim(encoding);
    bool isNarrowOne = narrowDim && narrowDim.size == 1;
    bool hasBatch = !cDims->batch.empty();

    // Saturn OPU narrow-M fallback: `iree_uk_opu_matmul` hangs on Saturn
    // for M < 128 (verified on vit_small dispatch 9, M=64). For these
    // shapes, use the CPU tile enumeration (smaller tiles, RVV-friendly)
    // instead of OPU 16×16 tiles. This produces a `linalg.mmt4d` with
    // CPU-style packing that flows through the RVV vectorization path,
    // sidestepping the OPU ukernel narrow-M hang. Batch matmuls
    // (attention QKV) still take the `hasBatch` branch below — their
    // working mechanism is inline VOPACC through VectorContractCustomKernels.

    // For batch matmuls and vecmat: use the same OPU 16x16 tiles as
    // standard matmuls (not identity, not CPU-specific tiles). This ensures
    // consistent tile sizes across LHS, RHS, and result operands, avoiding
    // batch_mmt4d shape mismatches. The batch dimension is preserved as-is
    // by getEncodingInfoForMatmul (only M, N, K dimensions get tiled).
    if (hasBatch) {
      // Batch matmuls: use OPU 16×16 tiles for all operands.
      // Skip narrowDim to keep tiles consistent across operands.
      // Prefer 32×32 when shapes allow — triggers sub32x32_full fast path
      // in iree_uk_opu_matmul (4 matrix registers used concurrently).
      // chooseMatmulTile falls back to 16×16 / 8×16 / 4×16 / 1×16 if the
      // shape doesn't accommodate the larger tile.
      SmallVector<TileMxNxK> opuTiles = {
          TileMxNxK{32, 32, 1},
          TileMxNxK{16, 16, 1},
          TileMxNxK{8, 16, 1},
          TileMxNxK{4, 16, 1},
          TileMxNxK{1, 16, 1},
      };
      // Do NOT use narrowDim for batch matmuls — it causes inconsistent
      // tile selection across operands, producing batch_mmt4d mismatches.
      IREE::Encoding::MatmulNarrowDim noNarrow;  // default: Dim::None
      TileMxNxK chosenTile = chooseMatmulTile(opuTiles, noNarrow);
      auto tileInfo = getEncodingInfoForMatmul(encoding, chosenTile);
      if (failed(tileInfo)) return info;
      return std::move(tileInfo.value());
    }
    if (isNarrowOne) {
      // Prefer 32×32 when shapes allow — triggers sub32x32_full fast path
      // in iree_uk_opu_matmul (4 matrix registers used concurrently).
      // chooseMatmulTile falls back to 16×16 / 8×16 / 4×16 / 1×16 if the
      // shape doesn't accommodate the larger tile.
      SmallVector<TileMxNxK> opuTiles = {
          TileMxNxK{32, 32, 1},
          TileMxNxK{16, 16, 1},
          TileMxNxK{8, 16, 1},
          TileMxNxK{4, 16, 1},
          TileMxNxK{1, 16, 1},
      };
      TileMxNxK chosenTile = chooseMatmulTile(opuTiles, narrowDim);
      auto tileInfo = getEncodingInfoForMatmul(encoding, chosenTile);
      if (failed(tileInfo)) return info;
      return std::move(tileInfo.value());
    }

    {
      // Standard matmul (M >= 2, N >= 2): use OPU efficient path.
      // RESULT: identity encoding (no packing, ukernel writes 2D directly).
      if (encoding.getOperandIndex().getValue() ==
          IREE::Encoding::MATMUL_RESULT) {
        return info; // Empty = identity layout
      }

      // LHS/RHS: OPU hardware-width tiles. Prefer 32×32 when shapes
      // allow — it triggers the sub32x32_full fast path in
      // iree_uk_opu_matmul (4 matrix registers used concurrently).
      // chooseMatmulTile falls back to 16×16 / 8×16 / 4×16 / 1×16 if
      // the shape doesn't accommodate the larger tile.
      // Gate 32×32 on iteration sizes: chooseMatmulTile only computes
      // paddingPenalty when narrowDim is set, so for a "balanced"
      // small matmul like 16×16×16 (mlp_wide) it would otherwise pick
      // 32×32 by volume and silently pad M to 32 → OOB writes / hang.
      // Only advertise tiles whose M ≤ iteration-M and N ≤ iteration-N.
      int64_t iterM = 0, iterN = 0;
      FailureOr<IREE::Encoding::BxMxNxKxKb> sizes =
          IREE::Encoding::getEncodingContractionLikeSizes(encoding);
      if (succeeded(sizes)) {
        iterM = sizes->M;
        iterN = sizes->N;
      }
      auto fits = [&](TileMxNxK t) {
        return (iterM == 0 || t.M <= iterM) && (iterN == 0 || t.N <= iterN);
      };
      SmallVector<TileMxNxK> opuTiles;
      for (TileMxNxK t : SmallVector<TileMxNxK>{
               TileMxNxK{32, 32, 1}, // OPU 32×32 fast path (M,N >= 32)
               TileMxNxK{16, 16, 1}, // OPU hardware tile
               TileMxNxK{8, 16, 1},  // Narrow-M
               TileMxNxK{4, 16, 1},  // Narrower-M
               TileMxNxK{1, 16, 1},  // Vecmat
           }) {
        if (fits(t)) opuTiles.push_back(t);
      }
      if (opuTiles.empty()) opuTiles.push_back(TileMxNxK{1, 16, 1});
      TileMxNxK chosenTileMxNxK = chooseMatmulTile(opuTiles, narrowDim);
      FailureOr<MaterializeEncodingInfo> maybeEncodingInfo =
          getEncodingInfoForMatmul(encoding, chosenTileMxNxK);
      if (failed(maybeEncodingInfo)) {
        return info;
      }
      return std::move(maybeEncodingInfo.value());
    }

    // Batch matmul or vecmat/matvec: fall back to CPU tile enumeration.
    // CPU tiles properly handle batch dims and K > 1 for non-trivial packing.
    SmallVector<TileMxNxK> cpuTiles =
        enumerateCPUMatmulTiles(encoding, layoutAttr.getConfiguration());
    if (cpuTiles.empty()) {
      return info;
    }
    TileMxNxK cpuTile = chooseMatmulTile(cpuTiles, narrowDim);
    FailureOr<MaterializeEncodingInfo> cpuInfo =
        getEncodingInfoForMatmul(encoding, cpuTile);
    if (failed(cpuInfo)) {
      return info;
    }
    return std::move(cpuInfo.value());
  }
};

struct OPUEncodingResolverMaterializerAttr final
    : EncodingLayoutMaterializerAttrExternalModelBase<
          OPUEncodingResolverMaterializerAttr, OPUEncodingResolverAttr> {

  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) {
      return nullptr;
    }
    if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      return lowerFillOpWithResolvedLayouts(b, fillOp, convertedResTypes,
                                            convertedOperands);
    }
    if (linalg::isaContractionOpInterface(linalgOp)) {
      // Try OPU ukernel lowering first. If it returns null (e.g., unsupported
      // shape or ukernels disabled), fall through to the standard CPU
      // contraction lowering with mmt4d.
      if (auto *ukOp =
              lowerOPUContractionToUkernel(b, linalgOp, convertedOperands)) {
        return ukOp;
      }
      if (auto *cpuOp = lowerContractionOpWithEncoding(
              b, linalgOp, convertedOperands,
              cast<IREE::Encoding::LayoutMaterializerAttr>(attr))) {
        return cpuOp;
      }
      // If both OPU and CPU contraction lowering fail, drop encoding.
      int64_t numInputs = linalgOp.getNumDpsInputs();
      int64_t numInits = linalgOp.getNumDpsInits();
      if (numInputs + numInits == (int64_t)convertedOperands.size()) {
        return dropEncodingAndCloneOp(b, op,
                                      convertedOperands.take_front(numInputs),
                                      convertedOperands.drop_front(numInputs));
      }
      // Operand count mismatch (e.g., im2col-generated shapes). Use resilient
      // fallback that clones with converted operands/types directly.
      return dropAllEncodingsAndClone(b, op, convertedOperands,
                                      convertedResTypes);
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      if (auto *lowered = lowerGenericOpWithResolvedLayouts(
              b, genericOp, convertedResTypes, convertedOperands,
              cast<IREE::Encoding::LayoutMaterializerAttr>(attr))) {
        return lowered;
      }
      // Fallback for elementwise generics that carry encoding attributes
      // (e.g. dequant/requant/bias-add fused with matmul in QDQ models).
      // Drop encoding and clone as-is.
      int64_t numInputs = linalgOp.getNumDpsInputs();
      int64_t numInits = linalgOp.getNumDpsInits();
      if (numInputs + numInits == (int64_t)convertedOperands.size()) {
        return dropEncodingAndCloneOp(b, op,
                                      convertedOperands.take_front(numInputs),
                                      convertedOperands.drop_front(numInputs));
      }
      // Operand count mismatch: resilient fallback.
      return dropAllEncodingsAndClone(b, op, convertedOperands,
                                      convertedResTypes);
    }
    // Catch-all: never return nullptr from the materializer.
    return dropAllEncodingsAndClone(b, op, convertedOperands,
                                    convertedResTypes);
  }
};

struct OPULayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<
          OPULayoutResolverAttr, OPUEncodingResolverAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    MLIRContext *ctx = attr.getContext();
    SmallVector<NamedAttribute> configItems;
    if (std::optional<StringRef> cpuFeatures = getConfigCpuFeatures(config)) {
      addConfigCpuFeatures(ctx, cpuFeatures.value(), configItems);
    }
    if (std::optional<StringRef> targetTriple = getConfigTargetTriple(config)) {
      addConfigTargetTriple(ctx, targetTriple.value(), configItems);
    }
    storeNamedAttrIfPresent(configItems, config, "ukernels");
    return OPUEncodingResolverAttr::get(ctx,
                                        DictionaryAttr::get(ctx, configItems));
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    MLIRContext *ctx = attr.getContext();
    return OPUEncodingResolverAttr::get(ctx, getPackedLayoutImpl(attr, type));
  }
};

struct OPUSerializableAttr final
    : IREE::Encoding::SerializableAttr::ExternalModel<OPUSerializableAttr,
                                                      OPUEncodingResolverAttr> {
  bool isSerialized(Attribute attr) const {
    auto configuration =
        cast<OPUEncodingResolverAttr>(attr).getConfiguration();
    return configuration && configuration.contains(kEncodingInfoAttrName);
  }

  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    return calculatePackedStorageSizeInBytesImpl(attr, loc, builder, type,
                                                 dynamicDims);
  }
};

struct OPUEncodingResolverVerifier
    : mlir::VerifiableTensorEncoding::ExternalModel<OPUEncodingResolverVerifier,
                                                    OPUEncodingResolverAttr> {
  LogicalResult
  verifyEncoding(Attribute attr, ArrayRef<int64_t> shape, Type elementType,
                 function_ref<InFlightDiagnostic()> emitError) const {
    auto packedLayoutMaterializerAttr =
        cast<Codegen::PackedLayoutMaterializerAttr>(attr);
    return packedLayoutMaterializerAttr.verifyPackedLayoutWithType(
        shape, elementType, emitError);
  }
};
// ===== Saturn OPU (+xopu) END =====

} // namespace

void registerCPUEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::CPU::IREECPUDialect *dialect) {
        IREE::CPU::CPUEncodingResolverAttr::attachInterface<
            CPUEncodingPackedLayoutMaterializerAttr,
            CPUEncodingResolverMaterializerAttr, CPULayoutResolverAttr,
            CPUSerializableAttr, CPUEncodingResolverVerifier>(*ctx);
        IREE::CPU::VMVXEncodingResolverAttr::attachInterface<
            VMVXEncodingPackedLayoutMaterializerAttr,
            VMVXEncodingResolverMaterializerAttr, VMVXLayoutResolverAttr,
            VMVXSerializableAttr, VMVXEncodingResolverVerifier>(*ctx);
        IREE::CPU::OPUEncodingResolverAttr::attachInterface<
            OPUEncodingPackedLayoutMaterializerAttr,
            OPUEncodingResolverMaterializerAttr, OPULayoutResolverAttr,
            OPUSerializableAttr, OPUEncodingResolverVerifier>(*ctx);
      });
}

} // namespace mlir::iree_compiler::IREE::CPU
