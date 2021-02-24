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

#include "Dyn/Dialect/IRDL/TypeConstraint.h"
#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"

using namespace mlir;
using namespace dyn;
using namespace irdl;

FailureOr<EqTypeConstraint>
EqTypeConstraint::get(StringRef typeName, OperationOp op, DynamicContext &ctx) {
  auto dialectEnd = typeName.find('.');

  StringRef dialectName, typeSubname;
  if (dialectEnd == std::string::npos) {
    dialectName = op.getDialectOp().name();
    typeSubname = typeName;
  } else {
    dialectName = StringRef(typeName).substr(0, dialectEnd);
    typeSubname = StringRef(typeName).substr(dialectEnd + 1);
  }

  /// Get the dialect owning the type.
  auto dialectRes = ctx.lookupDialect(dialectName);
  if (failed(dialectRes))
    return LogicalResult(
        op->emitOpError("dialect ").append(dialectName, " is not registered."));
  auto *dialect = *dialectRes;

  /// Get the type from the dialect.
  auto dynTypeRes = dialect->lookupType(typeSubname);
  if (failed(dynTypeRes))
    return LogicalResult(op->emitOpError("type ").append(
        typeSubname, " is not registered in the dialect ", dialectName, "."));
  auto *dynType = *dynTypeRes;

  auto type = DynamicType::get(ctx.getMLIRCtx(), dynType);

  return EqTypeConstraint(type);
}

LogicalResult EqTypeConstraint::verifyType(Operation *op, Type argType,
                                           bool isOperand, unsigned pos,
                                           dyn::DynamicContext &ctx) {
  if (type == this->type)
    return success();

  auto argCategory = isOperand ? "operand" : "result";

  return op->emitOpError("#").append(pos, " ", argCategory,
                                     " must be of type '", type, "', but got ",
                                     argType);
}
