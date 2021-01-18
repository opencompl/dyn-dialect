//===- DynDialect.cpp - Dyn dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynDialect.h"
#include "Dyn/DynOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::dyn;

DynDialect::DynDialect(mlir::MLIRContext *ctx)
  : Dialect(getDialectNamespace(), ctx, DynDialect::getTypeID()) {
  addOperations<DynOp>();
}
