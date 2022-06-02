//===- IRDL2SSA.h - Register dialects defined in IRDL -----------*- C++ -*-===//
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

#ifndef DYNOPT_IRDL2SSA_H
#define DYNOPT_IRDL2SSA_H

#include "Dyn/Dialect/IRDL/IRDLContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

namespace irdl2ssa {

/// Converts IRDL dialect definitions to IRDL-SSA dialect definitions.
/// This pass takes a type context as parameter information on
/// available types in the context.
class IRDL2SSA
    : public mlir::PassWrapper<IRDL2SSA, mlir::OperationPass<mlir::ModuleOp>> {
  mlir::irdl::TypeContext typeCtx;

public:
  IRDL2SSA(mlir::irdl::TypeContext ctx) : typeCtx(ctx) {}

  void runOnOperation() override;

  mlir::StringRef getArgument() const final { return "irdl2ssa"; }
};

} // namespace irdl2ssa

#endif // DYNOPT_IRDL2SSA_H
