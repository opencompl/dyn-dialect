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
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.h"
#include "mlir/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::irdlssa;

namespace mlir {
namespace irdl {

static mlir::Value
registerConstraintAsSSA(Attribute attr, TypeContext &typeCtx,
                        mlir::ConversionPatternRewriter &rewriter,
                        SmallVector<std::pair<StringRef, Value>> &vars,
                        mlir::Location location) {
  if (auto eqAttr = attr.dyn_cast<EqTypeConstraintAttr>()) {
    irdlssa::SSA_IsType op = rewriter.create<irdlssa::SSA_IsType>(
        location, rewriter.getType<irdlssa::ConstraintType>(),
        ParamTypeAttrOrAnyAttr::get(rewriter.getContext(),
                                    TypeAttr::get(eqAttr.getType())));
    return op.getResult();
  }

  if (auto anyAttr = attr.dyn_cast<AnyTypeConstraintAttr>()) {
    irdlssa::SSA_AnyType op = rewriter.create<irdlssa::SSA_AnyType>(
        location, rewriter.getType<irdlssa::ConstraintType>());
    return op.getResult();
  }

  if (auto anyOfAttr = attr.dyn_cast<AnyOfTypeConstraintAttr>()) {
    SmallVector<Value> constrs;
    for (auto constrAttr : anyOfAttr.getConstrs()) {
      constrs.push_back(registerConstraintAsSSA(constrAttr, typeCtx, rewriter,
                                                vars, location));
    }

    irdlssa::SSA_AnyOf op = rewriter.create<irdlssa::SSA_AnyOf>(
        location, rewriter.getType<irdlssa::ConstraintType>(), constrs);
    return op.getResult();
  }

  if (auto andAttr = attr.dyn_cast<AndTypeConstraintAttr>()) {
    SmallVector<Value> constrs;
    for (auto constrAttr : andAttr.getConstrs()) {
      constrs.push_back(registerConstraintAsSSA(constrAttr, typeCtx, rewriter,
                                                vars, location));
    }

    irdlssa::SSA_And op = rewriter.create<irdlssa::SSA_And>(
        location, rewriter.getType<irdlssa::ConstraintType>(), constrs);
    return op.getResult();
  }

  if (auto varAttr = attr.dyn_cast<VarTypeConstraintAttr>()) {
    for (auto var : vars) {
      if (var.first == varAttr.getName()) {
        return var.second;
      }
    }
    assert(false && "Unknown variable constraint");
  }

  if (auto dynBaseAttr = attr.dyn_cast<DynTypeBaseConstraintAttr>()) {
    SmallVector<Value> constrs;
    auto typeInfo = typeCtx.types.find(dynBaseAttr.getTypeName());
    if (typeInfo != typeCtx.types.end()) {
      for (size_t i = 0; i < typeInfo->getValue().paramAmount; i++) {
        constrs.push_back(rewriter.create<irdlssa::SSA_AnyType>(
            location, rewriter.getType<irdlssa::ConstraintType>()));
      }
    }

    irdlssa::SSA_ParametricType op =
        rewriter.create<irdlssa::SSA_ParametricType>(
            location, rewriter.getType<irdlssa::ConstraintType>(),
            dynBaseAttr.getTypeName(), constrs);
    return op.getResult();
  }

  if (auto baseAttr = attr.dyn_cast<TypeBaseConstraintAttr>()) {
    SmallVector<Value> constrs;
    for (size_t i = 0; i < baseAttr.getTypeDef()->getParameterAmount(); i++) {
      constrs.push_back(rewriter.create<irdlssa::SSA_AnyType>(
          location, rewriter.getType<irdlssa::ConstraintType>()));
    }

    irdlssa::SSA_ParametricType op =
        rewriter.create<irdlssa::SSA_ParametricType>(
            location, rewriter.getType<irdlssa::ConstraintType>(),
            baseAttr.getTypeDef()->getName(), constrs);
    return op.getResult();
  }

  if (auto dynParamsAttr = attr.dyn_cast<DynTypeParamsConstraintAttr>()) {
    SmallVector<Value> constrs;
    for (auto constrAttr : dynParamsAttr.getParamConstraints()) {
      constrs.push_back(registerConstraintAsSSA(constrAttr, typeCtx, rewriter,
                                                vars, location));
    }

    irdlssa::SSA_ParametricType op =
        rewriter.create<irdlssa::SSA_ParametricType>(
            location, rewriter.getType<irdlssa::ConstraintType>(),
            dynParamsAttr.getTypeName(), constrs);
    return op.getResult();
  }

  if (auto paramsAttr = attr.dyn_cast<TypeParamsConstraintAttr>()) {
    SmallVector<Value> constrs;
    for (auto constrAttr : paramsAttr.getParamConstraints()) {
      constrs.push_back(registerConstraintAsSSA(constrAttr, typeCtx, rewriter,
                                                vars, location));
    }

    irdlssa::SSA_ParametricType op =
        rewriter.create<irdlssa::SSA_ParametricType>(
            location, rewriter.getType<irdlssa::ConstraintType>(),
            paramsAttr.getTypeDef()->getName(), constrs);
    return op.getResult();
  }

  assert(false && "Unknown Type constraint");
}

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
        auto varConstr = var.getConstraint();
        Value varVal = registerConstraintAsSSA(
            varConstr, this->typeContext, rewriter, vars, constVars->getLoc());
        vars.push_back({var.getName(), varVal});
      }
      rewriter.eraseOp(constVars);
    });

    r.walk([&](ParametersOp paramOp) {
      SmallVector<Value> params;
      for (auto param : paramOp.getParams()) {
        auto constr = param.cast<NamedTypeConstraintAttr>().getConstraint();
        params.push_back(registerConstraintAsSSA(
            constr, this->typeContext, rewriter, vars, paramOp->getLoc()));
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
        auto varConstr = var.getConstraint();
        Value varVal = registerConstraintAsSSA(
            varConstr, this->typeContext, rewriter, vars, constVars->getLoc());
        vars.push_back({var.getName(), varVal});
      }
      rewriter.eraseOp(constVars);
    });

    r.walk([&](OperandsOp paramOp) {
      SmallVector<Value> params;
      for (auto param : paramOp.getParams()) {
        auto constr = param.cast<NamedTypeConstraintAttr>().getConstraint();
        params.push_back(registerConstraintAsSSA(
            constr, this->typeContext, rewriter, vars, paramOp->getLoc()));
      }

      rewriter.replaceOpWithNewOp<SSA_OperandsOp>(paramOp.getOperation(),
                                                  params);
    });

    r.walk([&](ResultsOp resultOp) {
      SmallVector<Value> results;
      for (auto result : resultOp.getParams()) {
        auto constr = result.cast<NamedTypeConstraintAttr>().getConstraint();
        results.push_back(registerConstraintAsSSA(
            constr, this->typeContext, rewriter, vars, resultOp->getLoc()));
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
