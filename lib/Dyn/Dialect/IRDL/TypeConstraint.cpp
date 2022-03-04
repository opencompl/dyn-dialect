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

LogicalResult EqTypeConstraint::verifyType(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
    MutableArrayRef<Type> varsValue) {
  if (type == expectedType)
    return success();

  if (emitError)
    return (*emitError)().append("expected type ", expectedType,
                                 " but got type ", type);
  return failure();
}

LogicalResult AnyOfTypeConstraint::verifyType(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
    MutableArrayRef<Type> varsValue) {
  SmallVector<Type> newVarsValue(varsValue.begin(), varsValue.end());
  for (auto &constr : constrs) {
    if (succeeded(
            constr->verifyType({}, type, typeConstraintVars, varsValue))) {
      for (size_t i = 0; i < varsValue.size(); i++)
        varsValue[i] = newVarsValue[i];
      return success();
    }
    for (size_t i = 0; i < varsValue.size(); i++)
      newVarsValue[i] = varsValue[i];
  }

  if (emitError)
    return (*emitError)().append("type ", type,
                                 " does not satisfy the constraint");
  return failure();
}

LogicalResult VarTypeConstraint::verifyType(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
    MutableArrayRef<Type> varsValue) {
  assert(varIndex < typeConstraintVars.size() &&
         "type constraint variable index out of bounds");
  assert(typeConstraintVars.size() == varsValue.size() &&
         "the number of constraints variables should be equal to the number of "
         "constraint variable values");

  // We first check if the variable was already assigned.
  auto expectedType = varsValue[varIndex];
  if (expectedType) {
    // If it is assigned, we check that our type is equal. If it is, we already
    // know we satisfy the underlying constraint.
    if (type == expectedType) {
      return success();
    } else {
      if (emitError)
        return (*emitError)().append("expected ", expectedType, " but got ",
                                     type);
      return failure();
    }
  }

  // We check that the type satisfies the type variable.
  if (failed(typeConstraintVars[varIndex]->verifyType(
          emitError, type, typeConstraintVars, varsValue)))
    return failure();

  // We assign the variable
  varsValue[varIndex] = type;

  return success();
}

LogicalResult DynTypeBaseConstraint::verifyType(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
    MutableArrayRef<Type> varsValue) {
  auto dynType = type.dyn_cast<DynamicType>();
  if (!dynType || dynType.getTypeDef() != dynTypeDef) {
    if (emitError)
      return (*emitError)().append(
          "expected base type '", dynTypeDef->getDialect()->getNamespace(), ".",
          dynTypeDef->getName(), "' but got type ", type);
    return failure();
  }
  return success();
}

LogicalResult DynTypeParamsConstraint::verifyType(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
    MutableArrayRef<Type> varsValue) {
  auto dynType = type.dyn_cast<DynamicType>();
  if (!dynType || dynType.getTypeDef() != dynTypeDef) {
    if (emitError)
      return (*emitError)().append(
          "expected base type '", dynTypeDef->getDialect()->getNamespace(), ".",
          dynTypeDef->getName(), "' but got type ", type);
    return failure();
  }

  // Since we do not have variadic parameters yet, we should have the
  // exact number of constraints.
  assert(dynType.getParams().size() == paramConstraints.size() &&
         "unexpected number of parameters in parameter type constraint");
  auto params = dynType.getParams();
  for (size_t i = 0; i < params.size(); i++) {
    auto paramType = params[i].cast<TypeAttr>().getValue();
    if (failed(paramConstraints[i]->verifyType(emitError, paramType,
                                               typeConstraintVars, varsValue)))
      return failure();
  }

  return success();
}

LogicalResult TypeParamsConstraint::verifyType(
    Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
    ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
    MutableArrayRef<Type> varsValue) {
  if (!typeDef->isCorrectType(type)) {
    if (emitError)
      return (*emitError)().append("expected base type '", typeDef->getName(),
                                   "' but got type ", type);
    return failure();
  }

  auto params = typeDef->getParameters(type);
  // Since we do not have variadic parameters yet, we should have the
  // exact number of constraints.
  assert(params.size() == paramConstraints.size() &&
         "unexpected number of parameters in parameter type constraint");
  for (size_t i = 0; i < params.size(); i++) {
    auto paramType = params[i].cast<TypeAttr>().getValue();
    if (failed(paramConstraints[i]->verifyType(emitError, paramType,
                                               typeConstraintVars, varsValue)))
      return failure();
  }

  return success();
}
