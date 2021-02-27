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

LogicalResult EqTypeConstraint::verifyType(Operation *op, Type argType,
                                           bool isOperand, unsigned pos,
                                           dyn::DynamicContext &ctx) {
  if (type == argType)
    return success();

  auto argCategory = isOperand ? "operand" : "result";

  return op->emitOpError("#").append(pos, " ", argCategory,
                                     " must be of type '", type, "', but got ",
                                     argType);
}

LogicalResult AnyOfTypeConstraint::verifyType(Operation *op, Type argType,
                                              bool isOperand, unsigned pos,
                                              dyn::DynamicContext &ctx) {
  if (std::find(types.begin(), types.end(), argType) != types.end())
    return success();

  auto argCategory = isOperand ? "operand" : "result";

  auto diagnostic = op->emitOpError("#").append(
      pos, " ", argCategory, " is of type ", argType,
      " but is expected to be one of the following types: [");

  for (size_t i = 0; i + 1 < types.size(); i++) {
    diagnostic.append(types[i], ", ");
  }
  return diagnostic.append(types.back(), "]");
}
