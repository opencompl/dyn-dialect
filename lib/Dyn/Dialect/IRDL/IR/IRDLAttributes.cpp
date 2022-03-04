//===- IRDLAttributes.h - Attributes definition for IRDL --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes used in the IRDL dialect.
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "Dyn/Dialect/IRDL/TypeConstraint.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Metadata.h"

#define GET_ATTRDEF_CLASSES
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.cpp.inc"

using namespace mlir;
using namespace irdl;

namespace mlir {
namespace irdl {

void IRDLDialect::registerAttributes() {
#define GET_ATTRDEF_LIST
  addAttributes<
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.cpp.inc"
      >();
}

} // namespace irdl
} // namespace mlir

//===----------------------------------------------------------------------===//
// IRDL equality type constraint attribute
//===----------------------------------------------------------------------===//

std::unique_ptr<TypeConstraint> EqTypeConstraintAttr::getTypeConstraint(
    SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const
        &constrVars) {
  return std::make_unique<EqTypeConstraint>(getType());
}

//===----------------------------------------------------------------------===//
// Always true type constraint attribute
//===----------------------------------------------------------------------===//

std::unique_ptr<TypeConstraint> AnyTypeConstraintAttr::getTypeConstraint(
    SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const
        &constrVars) {
  return std::make_unique<AnyTypeConstraint>();
}

//===----------------------------------------------------------------------===//
// IRDL AnyOf type constraint attribute
//===----------------------------------------------------------------------===//

std::unique_ptr<TypeConstraint> AnyOfTypeConstraintAttr::getTypeConstraint(
    SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const
        &constrVars) {
  SmallVector<std::unique_ptr<TypeConstraint>> constraints;
  auto constraintAttrs = getConstrs();
  for (auto constrAttr : constraintAttrs)
    constraints.push_back(
        constrAttr.cast<TypeConstraintAttrInterface>().getTypeConstraint(
            constrVars));
  return std::make_unique<AnyOfTypeConstraint>(std::move(constraints));
}

//===----------------------------------------------------------------------===//
// Type constraint variable
//===----------------------------------------------------------------------===//

std::unique_ptr<TypeConstraint> VarTypeConstraintAttr::getTypeConstraint(
    SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const
        &constrVars) {
  auto name = getName();
  // Iterate in reverse to match the latest defined variable.
  for (int i = constrVars.size() - 1; i >= 0; i--) {
    if (constrVars[i].first == name) {
      return std::make_unique<VarTypeConstraint>(i);
    }
  }
  // TODO: Make this an error
  assert(false && "Unknown type constraint variable");
}

//===----------------------------------------------------------------------===//
// Attribute for constraint on dynamic type base type
//===----------------------------------------------------------------------===//

std::unique_ptr<TypeConstraint> DynTypeBaseConstraintAttr::getTypeConstraint(
    SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const
        &constrVars) {
  auto splittedTypeName = getTypeName().split('.');
  auto dialectName = splittedTypeName.first;
  auto typeName = splittedTypeName.second;

  auto dialect = getContext()->getOrLoadDialect(dialectName);
  assert(dialect && "dialect is not registered");
  auto extensibleDialect = llvm::dyn_cast<ExtensibleDialect>(dialect);
  assert(extensibleDialect && "dialect is not extensible");

  auto *typeDef = extensibleDialect->lookupTypeDefinition(typeName);

  return std::make_unique<DynTypeBaseConstraint>(typeDef);
}

//===----------------------------------------------------------------------===//
// Attribute for constraint on dynamic type parameters
//===----------------------------------------------------------------------===//

std::unique_ptr<TypeConstraint> DynTypeParamsConstraintAttr::getTypeConstraint(
    SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const
        &constrVars) {
  SmallVector<std::unique_ptr<TypeConstraint>> paramConstraints;
  for (auto paramConstraintAttr : getParamConstraints())
    paramConstraints.push_back(
        paramConstraintAttr.cast<TypeConstraintAttrInterface>()
            .getTypeConstraint(constrVars));

  auto splittedTypeName = getTypeName().split('.');
  auto dialectName = splittedTypeName.first;
  auto typeName = splittedTypeName.second;

  auto dialect = getContext()->getOrLoadDialect(dialectName);
  assert(dialect && "dialect is not registered");
  auto extensibleDialect = llvm::dyn_cast<ExtensibleDialect>(dialect);
  assert(extensibleDialect && "dialect is not extensible");

  auto *typeDef = extensibleDialect->lookupTypeDefinition(typeName);

  return std::make_unique<DynTypeParamsConstraint>(typeDef,
                                                   std::move(paramConstraints));
}

//===----------------------------------------------------------------------===//
// Attribute for constraint on non-dynamic type parameters
//===----------------------------------------------------------------------===//

std::unique_ptr<TypeConstraint> TypeParamsConstraintAttr::getTypeConstraint(
    SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const
        &constrVars) {
  SmallVector<std::unique_ptr<TypeConstraint>> paramConstraints;
  for (auto paramConstraintAttr : getParamConstraints())
    paramConstraints.push_back(
        paramConstraintAttr.cast<TypeConstraintAttrInterface>()
            .getTypeConstraint(constrVars));

  return std::make_unique<TypeParamsConstraint>(getTypeDef(),
                                                std::move(paramConstraints));
}
