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

  return emitError().append("expected type ", expectedType, " but got type ",
                            type);
}

LogicalResult
AnyOfTypeConstraint::verifyType(function_ref<InFlightDiagnostic()> emitError,
                                Type type) {
  if (std::find(types.begin(), types.end(), type) != types.end())
    return success();

  return emitError().append("type ", type, " does not satisfy the constraint");
}

LogicalResult DynTypeParamsConstraint::verifyType(
    function_ref<InFlightDiagnostic()> emitError, Type type) {
  auto dynType = type.dyn_cast<DynamicType>();
  if (!dynType || dynType.getTypeDef() != dynTypeDef)
    return emitError().append("expected base type ", dynTypeDef->getName(),
                              " but got type ", type);

  // Since we do not have variadic parameters yet, we should have the
  // exact number of constraints.
  assert(dynType.getParams().size() == paramConstraints.size() &&
         "unexpected number of parameters in parameter type constraint");
  auto params = dynType.getParams();
  for (size_t i = 0; i < params.size(); i++) {
    auto paramType = params[i].cast<TypeAttr>().getValue();
    if (failed(paramConstraints[i]->verifyType(emitError, paramType)))
      return failure();
  }

  return success();
}
