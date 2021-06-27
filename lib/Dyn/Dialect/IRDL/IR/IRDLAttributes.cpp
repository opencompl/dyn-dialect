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
#include "Dyn/Dialect/IRDL/TypeConstraint.h"
#include "llvm/ADT/STLExtras.h"

#define GET_ATTRDEF_CLASSES
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.cpp.inc"

using namespace mlir;
using namespace irdl;

namespace mlir {
namespace irdl {

llvm::hash_code hash_value(mlir::irdl::OpDef opDef) {
  return llvm::hash_combine(opDef.operandDef, opDef.resultDef, opDef.traitDefs);
}

llvm::hash_code hash_value(mlir::irdl::TypeDef typeDef) {
  return llvm::hash_combine(typeDef.name, typeDef.paramDefs);
}

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

std::unique_ptr<TypeConstraint> EqTypeConstraintAttr::getTypeConstraint() {
  return std::make_unique<EqTypeConstraint>(getType());
}

//===----------------------------------------------------------------------===//
// IRDL AnyOf type constraint attribute
//===----------------------------------------------------------------------===//

std::unique_ptr<TypeConstraint> AnyOfTypeConstraintAttr::getTypeConstraint() {
  return std::make_unique<AnyOfTypeConstraint>(getTypes());
}

//===----------------------------------------------------------------------===//
// Always true type constraint attribute
//===----------------------------------------------------------------------===//

std::unique_ptr<TypeConstraint> AnyTypeConstraintAttr::getTypeConstraint() {
  return std::make_unique<AnyTypeConstraint>();
}

//===----------------------------------------------------------------------===//
// Attribute for constraint on dynamic type parameters
//===----------------------------------------------------------------------===//

std::unique_ptr<TypeConstraint>
DynTypeParamsConstraintAttr::getTypeConstraint() {
  auto allEqs = llvm::all_of(getParamConstraints(), [](Attribute attr) {
    return attr.isa<EqTypeConstraintAttr>();
  });

  // If all parameter constraints are equalities, we can return an equality
  // type constraint
  if (allEqs) {
    SmallVector<Attribute> params;
    // Get all parameters from the equality constraints
    for (auto paramConstraintAttr : getParamConstraints())
      params.push_back(TypeAttr::get(
          paramConstraintAttr.cast<EqTypeConstraintAttr>().getType()));

    return std::make_unique<EqTypeConstraint>(
        DynamicType::get(getTypeDef(), params));
  }

  SmallVector<std::unique_ptr<TypeConstraint>> paramConstraints;
  for (auto paramConstraintAttr : getParamConstraints())
    paramConstraints.push_back(
        paramConstraintAttr.cast<TypeConstraintAttrInterface>()
            .getTypeConstraint());

  return std::make_unique<DynTypeParamsConstraint>(getTypeDef(),
                                                   std::move(paramConstraints));
}
