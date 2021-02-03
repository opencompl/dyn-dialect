//===- IRDLRegistration.h - IRDL registration -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the registration of MLIR objects from IRDL operations.
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IRDLRegistration.h"
#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace irdl;

namespace {
/// Register a type represented by a `irdl.type` operation.
LogicalResult registerType(TypeOp typeOp, dyn::DynamicDialect *dialect) {
  return dialect->createAndAddType(typeOp.name());
}

/// Register an operation represented by a `irdl.operation` operation.
LogicalResult registerOperation(OperationOp op, dyn::DynamicDialect *dialect) {
  return dialect->createAndAddOperation(op.name());
}
} // namespace

LogicalResult mlir::irdl::registerDialect(DialectOp dialectOp,
                                          dyn::DynamicContext *ctx) {

  // Register the dialect.
  auto dialectRes = ctx->createAndRegisterDialect(dialectOp.name());
  if (failed(dialectRes))
    return failure();
  auto *dialect = *dialectRes;

  // Register all the types first.
  auto failedTypeRegistration = dialectOp.walk([&](TypeOp type) {
    if (failed(registerType(type, dialect)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  // If a type failed to register, return early with an error.
  if (failedTypeRegistration.wasInterrupted())
    return failure();

  // Register the operations.
  auto failedOperationRegistration = dialectOp.walk([&](OperationOp op) {
    if (failed(registerOperation(op, dialect)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  return failure(failedOperationRegistration.wasInterrupted());
}
