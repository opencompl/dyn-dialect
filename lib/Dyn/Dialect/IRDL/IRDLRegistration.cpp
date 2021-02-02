//===- IRDLRegistration.h - IRDL registration -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the registration of IRDL-defined MLIR objects.
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IRDLRegistration.h"
#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace irdl;

namespace {
LogicalResult registerType(TypeOp typeOp, dyn::DynamicDialect *dialect) {
  return dialect->createAndAddType(typeOp.name());
}

LogicalResult registerOperation(OperationOp op, dyn::DynamicDialect *dialect) {
  return dialect->createAndAddOperation(op.name());
}
} // namespace

LogicalResult mlir::irdl::registerDialect(DialectOp dialectOp,
                                          dyn::DynamicContext *ctx) {

  // Register the dialect
  auto dialectRes = ctx->createAndRegisterDialect(dialectOp.name());
  if (failed(dialectRes))
    return failure();

  auto *dialect = *dialectRes;

  // Register the types
  auto failedTypeRegistration = success();
  dialectOp.walk([&](TypeOp type) {
    if (failed(registerType(type, dialect)))
      failedTypeRegistration = failure();
  });

  if (failed(failedTypeRegistration))
    return failure();

  // Register the operations
  auto failedOperationRegistration = success();
  dialectOp.walk([&](OperationOp op) {
    if (failed(registerOperation(op, dialect)))
      failedOperationRegistration = failure();
  });

  if (failed(failedOperationRegistration))
    return failure();

  return success();
}
