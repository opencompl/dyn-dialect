//===- IRDL2SSA.cpp - Register dialects defined in IRDL ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts IRDL dialect definitions to IRDL-SSA definitions.
//
//===----------------------------------------------------------------------===//

#include "IRDL2SSA.h"

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
#include "Dyn/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::irdl;
using namespace mlir::irdlssa;

namespace irdl2ssa {

// ./build/bin/dyn-opt test/Dyn/cmath.irdl --irdl2ssa
struct LowerIRDLDialect : public mlir::OpConversionPattern<DialectOp> {
  LowerIRDLDialect(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(DialectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto newOp = rewriter.create<SSA_DialectOp>(op.getLoc(), adaptor.name());
    rewriter.inlineRegionBefore(op.body(), newOp.body(), newOp.body().end());
    rewriter.eraseOp(op);

    return success();
  }
};

struct LowerIRDLType : public mlir::OpConversionPattern<TypeOp> {
  LowerIRDLType(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult match(TypeOp op) const override { return success(); }

  void rewrite(TypeOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    auto &r = op.getRegion();

    SmallVector<std::pair<StringRef, Value>> vars;
    rewriter.setInsertionPointToStart(&r.front());
    r.walk([&](ConstraintVarsOp constVars) {
      for (auto arg : constVars.params()) {
        auto var = arg.cast<NamedTypeConstraintAttr>();
        Value varVal = var.getConstraint()
                           .cast<TypeConstraintAttrInterface>()
                           .registerAsSSA(rewriter, vars, constVars->getLoc());
        vars.push_back({var.getName(), varVal});
      }
      rewriter.eraseOp(constVars);
    });

    r.walk([&](ParametersOp paramOp) {
      SmallVector<Value> params;
      for (auto param : paramOp.params()) {
        params.push_back(param.cast<NamedTypeConstraintAttr>()
                             .getConstraint()
                             .cast<TypeConstraintAttrInterface>()
                             .registerAsSSA(rewriter, vars, paramOp->getLoc()));
      }

      rewriter.replaceOpWithNewOp<SSA_ParametersOp>(paramOp.getOperation(),
                                                    params);
    });

    rewriter.setInsertionPoint(op);
    auto newOp = rewriter.create<SSA_TypeOp>(op.getLoc(), adaptor.name());
    rewriter.inlineRegionBefore(op.body(), newOp.body(), newOp.body().end());
    rewriter.eraseOp(op);
  }
};

struct LowerIRDLOp : public mlir::OpConversionPattern<OperationOp> {
  LowerIRDLOp(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult match(OperationOp op) const override { return success(); }

  void rewrite(OperationOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    auto &r = op.getRegion();

    SmallVector<std::pair<StringRef, Value>> vars;
    rewriter.setInsertionPointToStart(&r.front());
    r.walk([&](ConstraintVarsOp constVars) {
      for (auto arg : constVars.params()) {
        auto var = arg.cast<NamedTypeConstraintAttr>();
        Value varVal = var.getConstraint()
                           .cast<TypeConstraintAttrInterface>()
                           .registerAsSSA(rewriter, vars, constVars->getLoc());
        vars.push_back({var.getName(), varVal});
      }
      rewriter.eraseOp(constVars);
    });

    r.walk([&](OperandsOp paramOp) {
      SmallVector<Value> params;
      for (auto param : paramOp.params()) {
        params.push_back(param.cast<NamedTypeConstraintAttr>()
                             .getConstraint()
                             .cast<TypeConstraintAttrInterface>()
                             .registerAsSSA(rewriter, vars, paramOp->getLoc()));
      }

      rewriter.replaceOpWithNewOp<SSA_OperandsOp>(paramOp.getOperation(),
                                                  params);
    });

    r.walk([&](ResultsOp resultOp) {
      SmallVector<Value> results;
      for (auto result : resultOp.params()) {
        results.push_back(
            result.cast<NamedTypeConstraintAttr>()
                .getConstraint()
                .cast<TypeConstraintAttrInterface>()
                .registerAsSSA(rewriter, vars, resultOp->getLoc()));
      }

      rewriter.replaceOpWithNewOp<SSA_ResultsOp>(resultOp.getOperation(),
                                                 results);
    });

    rewriter.setInsertionPoint(op);
    auto newOp = rewriter.create<SSA_OperationOp>(op.getLoc(), adaptor.name());
    rewriter.inlineRegionBefore(op.body(), newOp.body(), newOp.body().end());
    rewriter.eraseOp(op);
  }
};

void IRDL2SSA::runOnOperation() {
  ConversionTarget target(this->getContext());

  target.addLegalDialect<IRDLSSADialect>();
  target.addIllegalDialect<IRDLDialect>();

  RewritePatternSet patterns(&this->getContext());
  patterns.insert<LowerIRDLDialect>(&this->getContext());
  patterns.insert<LowerIRDLType>(&this->getContext());
  patterns.insert<LowerIRDLOp>(&this->getContext());

  mlir::DenseSet<mlir::Operation *> unconverted;
  if (failed(mlir::applyPartialConversion(this->getOperation(), target,
                                          std::move(patterns), &unconverted))) {
    this->signalPassFailure();
  }
}

} // namespace irdl2ssa
