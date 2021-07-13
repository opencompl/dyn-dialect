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

#define GET_ATTRDEF_CLASSES
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.cpp.inc"

using namespace mlir;
using namespace irdl;

namespace mlir {
namespace irdl {

llvm::hash_code hash_value(mlir::irdl::OpTypeDef typeDef) {
  return llvm::hash_combine(typeDef.operandDef, typeDef.resultDef,
                            typeDef.traitDefs);
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
