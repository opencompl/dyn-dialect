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

using namespace mlir;
using namespace irdl;

LogicalResult mlir::irdl::registerDialect(DialectOp dialectOp,
                                          dyn::DynamicContext *ctx) {
  auto dialectRes = ctx->createAndRegisterDialect(dialectOp.name());
  if (failed(dialectRes))
    return failure();

  auto *dialect = *dialectRes;
  return dialect->createAndAddOperation("dummyop");
}
