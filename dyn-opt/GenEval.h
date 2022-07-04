//===- GenEval.h - Generates IRDL-Eval from IRDL-SSA ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates IRDL-Eval from IRDL-SSA definitions.
//
//===----------------------------------------------------------------------===//

#ifndef DYNOPT_GENEVAL_H
#define DYNOPT_GENEVAL_H

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace irdleval {

class GenEval
    : public mlir::PassWrapper<
          GenEval, mlir::OperationPass<mlir::irdlssa::SSA_DialectOp>> {

public:
  void runOnOperation() override;

  mlir::StringRef getArgument() const final { return "irdl-gen-eval"; }
};

} // namespace irdleval
} // namespace mlir

#endif // DYNOPT_GENEVAL_H
