//===- TypeConstraint.cpp - IRDL type constraint definition -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the different type constraints an operand or a result can
// have.
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/TypeConstraint.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"

using namespace mlir;
using namespace dyn;
using namespace irdl;

LogicalResult TypeConstraint::verifyType(Operation *op, Type type,
                                         bool isOperand, unsigned pos,
                                         dyn::DynamicContext &ctx) {
  auto dialectEnd = typeName.find('.');
  assert(dialectEnd != std::string::npos);
  auto dialectName = StringRef(typeName).substr(0, dialectEnd);
  auto typeSubname = StringRef(typeName).substr(dialectEnd + 1);

  /// Get the dialect owning the type.
  auto dialectRes = ctx.lookupDialect(dialectName);
  if (failed(dialectRes))
    return op->emitError("dialect " + dialectName + " is not registered.");
  auto *dialect = *dialectRes;

  /// Get the type from the dialect.
  auto dynTypeRes = dialect->lookupType(typeSubname);
  if (failed(dynTypeRes))
    return op->emitError("type " + typeSubname +
                         " is not registered in the dialect " + dialectName +
                         ".");
  auto *dynType = *dynTypeRes;

  /// Check for type equality
  if (!dyn::DynamicType::isa(type, dynType)) {
    auto argType = isOperand ? "operand" : "result";
    return op->emitError(std::to_string(pos) + "nth " + argType +
                         " should be of type " + typeName);
  }

  return success();
}
