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

using namespace mlir;
using namespace irdl;

LogicalResult
EqTypeConstraint::verifyType(function_ref<InFlightDiagnostic()> emitError,
                             Type type) {
  if (type == expectedType)
    return success();

  return emitError().append("must be of type '", expectedType,
                            "' but is of type", type);
}

LogicalResult
AnyOfTypeConstraint::verifyType(function_ref<InFlightDiagnostic()> emitError,
                                Type type) {
  if (std::find(types.begin(), types.end(), type) != types.end())
    return success();

  return emitError().append("invalid parameter type");
}
