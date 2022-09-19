//===- LowerIRDL.cpp - Translate IRDL to IRDL-SSA ---------------*- C++ -*-===//
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

#include "LowerIRDL.h"

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
#include "Dyn/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::irdlssa;

namespace mlir {
namespace irdl {

struct LowerIRDLDialect : public mlir::OpConversionPattern<DialectOp> {
  TypeContext &typeContext;

  LowerIRDLDialect(MLIRContext *context, TypeContext &typeContext)
      : OpConversionPattern(context), typeContext(typeContext) {}

  LogicalResult match(DialectOp op) const override { return success(); }

  void rewrite(DialectOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto newOp = rewriter.create<SSA_DialectOp>(op.getLoc(), adaptor.getName());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    rewriter.eraseOp(op);
  }
};

struct LowerIRDLType : public mlir::OpConversionPattern<TypeOp> {
  TypeContext &typeContext;

  LowerIRDLType(MLIRContext *context, TypeContext &typeContext)
      : OpConversionPattern(context), typeContext(typeContext) {}

  LogicalResult match(TypeOp op) const override { return success(); }

  void rewrite(TypeOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    auto &r = op.getBody();

    SmallVector<std::pair<StringRef, Value>> vars;
    rewriter.setInsertionPointToStart(&r.front());
    r.walk([&](ConstraintVarsOp constVars) {
      for (auto arg : constVars.getParams()) {
        auto var = arg.cast<NamedTypeConstraintAttr>();
        Value varVal = var.getConstraint()
                           .cast<TypeConstraintAttrInterface>()
                           .registerAsSSA(this->typeContext, rewriter, vars,
                                          constVars->getLoc());
        vars.push_back({var.getName(), varVal});
      }
      rewriter.eraseOp(constVars);
    });

    r.walk([&](ParametersOp paramOp) {
      SmallVector<Value> params;
      for (auto param : paramOp.getParams()) {
        params.push_back(param.cast<NamedTypeConstraintAttr>()
                             .getConstraint()
                             .cast<TypeConstraintAttrInterface>()
                             .registerAsSSA(this->typeContext, rewriter, vars,
                                            paramOp->getLoc()));
      }

      rewriter.replaceOpWithNewOp<SSA_ParametersOp>(paramOp.getOperation(),
                                                    params);
    });

    rewriter.setInsertionPoint(op);
    auto newOp = rewriter.create<SSA_TypeOp>(op.getLoc(), op.getName());
    rewriter.inlineRegionBefore(r, newOp.getBody(), newOp.getBody().end());
    rewriter.eraseOp(op);
  }
};

struct LowerIRDLOp : public mlir::OpConversionPattern<OperationOp> {
  TypeContext &typeContext;

  LowerIRDLOp(MLIRContext *context, TypeContext &typeContext)
      : OpConversionPattern(context), typeContext(typeContext) {}

  LogicalResult match(OperationOp op) const override { return success(); }

  void rewrite(OperationOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    auto &r = op.getBody();

    SmallVector<std::pair<StringRef, Value>> vars;
    rewriter.setInsertionPointToStart(&r.front());
    r.walk([&](ConstraintVarsOp constVars) {
      for (auto arg : constVars.getParams()) {
        auto var = arg.cast<NamedTypeConstraintAttr>();
        Value varVal = var.getConstraint()
                           .cast<TypeConstraintAttrInterface>()
                           .registerAsSSA(this->typeContext, rewriter, vars,
                                          constVars->getLoc());
        vars.push_back({var.getName(), varVal});
      }
      rewriter.eraseOp(constVars);
    });

    r.walk([&](OperandsOp paramOp) {
      SmallVector<Value> params;
      for (auto param : paramOp.getParams()) {
        params.push_back(param.cast<NamedTypeConstraintAttr>()
                             .getConstraint()
                             .cast<TypeConstraintAttrInterface>()
                             .registerAsSSA(this->typeContext, rewriter, vars,
                                            paramOp->getLoc()));
      }

      rewriter.replaceOpWithNewOp<SSA_OperandsOp>(paramOp.getOperation(),
                                                  params);
    });

    r.walk([&](ResultsOp resultOp) {
      SmallVector<Value> results;
      for (auto result : resultOp.getParams()) {
        results.push_back(result.cast<NamedTypeConstraintAttr>()
                              .getConstraint()
                              .cast<TypeConstraintAttrInterface>()
                              .registerAsSSA(this->typeContext, rewriter, vars,
                                             resultOp->getLoc()));
      }

      rewriter.replaceOpWithNewOp<SSA_ResultsOp>(resultOp.getOperation(),
                                                 results);
    });

    rewriter.setInsertionPoint(op);
    auto newOp = rewriter.create<SSA_OperationOp>(op.getLoc(), op.getName());
    rewriter.inlineRegionBefore(r, newOp.getBody(), newOp.getBody().end());
    rewriter.eraseOp(op);
  }
};

void LowerIRDL::runOnOperation() {
  ConversionTarget target(this->getContext());

  target.addLegalDialect<IRDLSSADialect>();
  target.addIllegalDialect<IRDLDialect>();

  ModuleOp op = this->getOperation();

  // Add locally-declared dynamic types to the type context
  op.walk([&](DialectOp d) {
    d.walk([&](TypeOp t) {
      t.walk([&](ParametersOp p) {
        std::string name;
        name.reserve(d.getName().size() + 1 + t.getName().size());
        name += d.getName();
        name += '.';
        name += t.getName();

        this->typeCtx.types.insert(
            {std::move(name), TypeContext::TypeInfo(p.getParams().size())});
      });
    });
  });

  // Apply the conversion
  RewritePatternSet patterns(&this->getContext());
  patterns.insert<LowerIRDLDialect>(&this->getContext(), this->typeCtx);
  patterns.insert<LowerIRDLType>(&this->getContext(), this->typeCtx);
  patterns.insert<LowerIRDLOp>(&this->getContext(), this->typeCtx);

  mlir::DenseSet<mlir::Operation *> unconverted;
  if (failed(mlir::applyPartialConversion(op, target, std::move(patterns),
                                          &unconverted))) {
    this->signalPassFailure();
  }
}

} // namespace irdl
} // namespace mlir
