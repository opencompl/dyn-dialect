//===- IRDLSSAOps.cpp - IRDL-SSA dialect ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

using namespace mlir;
using namespace mlir::irdlssa;

llvm::Optional<std::unique_ptr<TypeConstraint>>
SSA_IsType::getVerifier(SmallVector<Value> const &valueToConstr) {
  return {std::make_unique<IsTypeConstraint>(this->type())};
}

DynamicTypeDefinition *findDynamicType(SSA_ParametricType &op, StringRef type) {
  auto splitted = type.split('.');
  auto dialectName = splitted.first;
  auto typeName = splitted.second;

  auto dialect = op->getContext()->getOrLoadDialect(dialectName);
  if (!dialect)
    return nullptr;

  auto extensibleDialect = llvm::dyn_cast<ExtensibleDialect>(dialect);
  if (!extensibleDialect)
    return nullptr;

  return extensibleDialect->lookupTypeDefinition(typeName);
}

irdl::TypeWrapper *findTypeWrapper(SSA_ParametricType &op, StringRef type) {
  Dialect *irdlssaDialect = op.getContext()->getLoadedDialect("irdlssa");
  assert(irdlssaDialect && "irdlssa is not registered");

  IRDLSSADialect *irdlssa = dyn_cast<IRDLSSADialect>(irdlssaDialect);
  assert(irdlssa && "irdlssa dialect is not IRDL-SSA");

  return irdlssa->getTypeWrapper(type);
}

llvm::Optional<std::unique_ptr<TypeConstraint>>
SSA_ParametricType::getVerifier(SmallVector<Value> const &valueToConstr) {
  SmallVector<size_t> constraints;
  for (Value arg : this->args()) {
    for (size_t i = 0; i < valueToConstr.size(); i++) {
      if (valueToConstr[i] == arg) {
        constraints.push_back(i);
        break;
      }
    }
  }

  auto typeName = this->type();
  if (DynamicTypeDefinition *type = findDynamicType(*this, typeName)) {
    return {std::make_unique<DynParametricTypeConstraint>(
        type, std::move(constraints))};
  } else if (irdl::TypeWrapper *type = findTypeWrapper(*this, typeName)) {
    return {std::make_unique<ParametricTypeConstraint>(type,
                                                       std::move(constraints))};
  } else {
    this->emitError().append("type '", typeName,
                             "' is not declared at that point");
    return {};
  }
}

llvm::Optional<std::unique_ptr<TypeConstraint>>
SSA_AnyOf::getVerifier(SmallVector<Value> const &valueToConstr) {
  SmallVector<size_t> constraints;
  for (Value arg : this->args()) {
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
SSA_AnyType::getVerifier(SmallVector<Value> const &valueToConstr) {
  return {std::make_unique<AnyTypeConstraint>()};
}
