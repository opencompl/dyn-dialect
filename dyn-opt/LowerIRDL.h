//===- LowerIRDL.h - Translate IRDL to IRDL-SSA -----------------*- C++ -*-===//
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

#ifndef DYNOPT_LOWERIRDL_H
#define DYNOPT_LOWERIRDL_H

#include "Dyn/Dialect/IRDL/IRDLContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

namespace lowerirdl {

/// Converts IRDL dialect definitions to IRDL-SSA dialect definitions.
/// This pass takes a type context as parameter information on
/// available types in the context. It is not necessary to register
/// types declared within the translated IRDL declaration.
class LowerIRDL
    : public mlir::PassWrapper<LowerIRDL, mlir::OperationPass<mlir::ModuleOp>> {
  mlir::irdl::TypeContext typeCtx;

public:
  LowerIRDL(mlir::irdl::TypeContext ctx) : typeCtx(ctx) {}

  void runOnOperation() override;

  mlir::StringRef getArgument() const final { return "lower-irdl"; }
};

} // namespace irdl2ssa

#endif // DYNOPT_LOWERIRDL_H
