//===- IRDLSSAOps.cpp - IRDL-SSA dialect ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSAAttributes.h"
#include "mlir/Dialect/IRDL/TypeWrapper.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

using namespace mlir;
using namespace mlir::irdlssa;
using namespace mlir::irdl;

// Verifier instantiation
Attribute
instantiateParamType(llvm::function_ref<InFlightDiagnostic()> emitError,
                     MLIRContext &ctx, ParamTypeAttrOrAnyAttr attr) {
  if (ParamTypeInstanceAttr typeDesc =
          attr.getAttr().dyn_cast<ParamTypeInstanceAttr>()) {
    auto typeName = typeDesc.getBase();

    SmallVector<Attribute> params;
    for (ParamTypeAttrOrAnyAttr param : typeDesc.getParams()) {
      auto result = instantiateParamType(emitError, ctx, param);
      if (!result) {
        return Attribute();
      }
      params.push_back(result);
    }

    if (DynamicTypeDefinition *type = findDynamicType(ctx, typeName)) {
      DynamicType instantiated =
          DynamicType::getChecked(emitError, type, params);
      if (instantiated)
        return TypeAttr::get(instantiated);
      else
        return Attribute();
    } else if (TypeWrapper *type = findTypeWrapper(ctx, typeName)) {
      Type instantiated = type->instantiate(emitError, params);
      if (instantiated)
        return TypeAttr::get(instantiated);
      else
        return Attribute();
    } else {
      emitError().append("type ", typeName, " is not declared at that point");
      return {};
    }
  } else {
    return attr.getAttr();
  }
}

llvm::Optional<std::unique_ptr<TypeConstraint>>
SSA_IsType::getVerifier(SmallVector<Value> const &valueToConstr) {
  auto attr = instantiateParamType([&]() { return this->emitError(); },
                                   *this->getContext(), this->getExpected());

  if (!attr)
    return {};

  if (TypeAttr type = attr.dyn_cast<TypeAttr>()) {
    return {std::make_unique<IsTypeConstraint>(type.getValue())};
  } else {
    this->emitError().append("the provided attribute is not a type");
    return {};
  }
}

llvm::Optional<std::unique_ptr<TypeConstraint>>
SSA_ParametricType::getVerifier(SmallVector<Value> const &valueToConstr) {
  SmallVector<size_t> constraints;
  for (Value arg : this->getArgs()) {
    for (size_t i = 0; i < valueToConstr.size(); i++) {
      if (valueToConstr[i] == arg) {
        constraints.push_back(i);
        break;
      }
    }
  }

  auto typeName = this->getBaseType();
  if (DynamicTypeDefinition *type =
          findDynamicType(*this->getContext(), typeName)) {
    return {std::make_unique<DynParametricTypeConstraint>(
        type, std::move(constraints))};
  } else if (TypeWrapper *type =
                 findTypeWrapper(*this->getContext(), typeName)) {
    return {std::make_unique<ParametricTypeConstraint>(type,
                                                       std::move(constraints))};
  } else {
    this->emitError().append("type ", typeName,
                             " is not declared at that point");
    return {};
  }
}

llvm::Optional<std::unique_ptr<TypeConstraint>>
SSA_AnyOf::getVerifier(SmallVector<Value> const &valueToConstr) {
  SmallVector<size_t> constraints;
  for (Value arg : this->getArgs()) {
    for (size_t i = 0; i < valueToConstr.size(); i++) {
      if (valueToConstr[i] == arg) {
        constraints.push_back(i);
        break;
      }
    }
  }

  return {std::make_unique<AnyOfTypeConstraint>(constraints)};
}

llvm::Optional<std::unique_ptr<TypeConstraint>>
SSA_And::getVerifier(SmallVector<Value> const &valueToConstr) {
  SmallVector<size_t> constraints;
  for (Value arg : this->getArgs()) {
    for (size_t i = 0; i < valueToConstr.size(); i++) {
      if (valueToConstr[i] == arg) {
        constraints.push_back(i);
        break;
      }
    }
  }

  return {std::make_unique<AndTypeConstraint>(constraints)};
}

llvm::Optional<std::unique_ptr<TypeConstraint>>
SSA_AnyType::getVerifier(SmallVector<Value> const &valueToConstr) {
  return {std::make_unique<AnyTypeConstraint>()};
}
