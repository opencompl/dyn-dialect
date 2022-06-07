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
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "Dyn/Dialect/IRDL-SSA/IRDLSSAContext.h"
#include "Dyn/Dialect/IRDL-SSA/TypeWrapper.h"
#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Metadata.h"

//===----------------------------------------------------------------------===//
// TypeWrapper parameter
//===----------------------------------------------------------------------===//

using namespace mlir;
using namespace irdl;
using mlir::irdlssa::IRDLSSADialect;
using mlir::irdlssa::TypeContext;
using mlir::irdlssa::TypeWrapper;

namespace mlir {
AsmPrinter &operator<<(AsmPrinter &printer, TypeWrapper *param) {
  printer << param->getName();
  return printer;
}

template <> struct FieldParser<TypeWrapper *> {
  static FailureOr<TypeWrapper *> parse(AsmParser &parser) {
    std::string name;
    (void)parser.parseOptionalKeywordOrString(&name);
    auto *irdl = parser.getContext()->getOrLoadDialect<IRDLSSADialect>();
    auto typeWrapper = irdl->getTypeWrapper(name);
    if (!typeWrapper)
      return parser.emitError(parser.getCurrentLocation(), "Type wrapper ")
             << name << " was not registered in IRDL";
    return typeWrapper;
  }
};
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.cpp.inc"

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

mlir::Value EqTypeConstraintAttr::registerAsSSA(
    TypeContext &typeCtx, mlir::ConversionPatternRewriter &rewriter,
    SmallVector<std::pair<StringRef, Value>> &vars,
    mlir::Location location) const {
  irdlssa::SSA_IsType op = rewriter.create<irdlssa::SSA_IsType>(
      location, rewriter.getType<irdlssa::ConstraintType>(), this->getType());
  return op.getResult();
}

//===----------------------------------------------------------------------===//
// Always true type constraint attribute
//===----------------------------------------------------------------------===//

mlir::Value AnyTypeConstraintAttr::registerAsSSA(
    TypeContext &typeCtx, mlir::ConversionPatternRewriter &rewriter,
    SmallVector<std::pair<StringRef, Value>> &vars,
    mlir::Location location) const {
  irdlssa::SSA_AnyType op = rewriter.create<irdlssa::SSA_AnyType>(
      location, rewriter.getType<irdlssa::ConstraintType>());
  return op.getResult();
}

//===----------------------------------------------------------------------===//
// IRDL AnyOf type constraint attribute
//===----------------------------------------------------------------------===//

mlir::Value AnyOfTypeConstraintAttr::registerAsSSA(
    TypeContext &typeCtx, mlir::ConversionPatternRewriter &rewriter,
    SmallVector<std::pair<StringRef, Value>> &vars,
    mlir::Location location) const {
  SmallVector<Value> constrs;
  for (auto constrAttr : this->getConstrs()) {
    constrs.push_back(
        constrAttr.cast<TypeConstraintAttrInterface>().registerAsSSA(
            typeCtx, rewriter, vars, location));
  }

  irdlssa::SSA_AnyOf op = rewriter.create<irdlssa::SSA_AnyOf>(
      location, rewriter.getType<irdlssa::ConstraintType>(), constrs);
  return op.getResult();
}

//===----------------------------------------------------------------------===//
// Type constraint variable
//===----------------------------------------------------------------------===//

mlir::Value VarTypeConstraintAttr::registerAsSSA(
    TypeContext &typeCtx, mlir::ConversionPatternRewriter &rewriter,
    SmallVector<std::pair<StringRef, Value>> &vars,
    mlir::Location location) const {
  for (auto var : vars) {
    if (var.first == this->getName()) {
      return var.second;
    }
  }
  // TODO: Make this an error
  assert(false && "Unknown type constraint variable");
}

//===----------------------------------------------------------------------===//
// Attribute for constraint on dynamic type base type
//===----------------------------------------------------------------------===/

mlir::Value DynTypeBaseConstraintAttr::registerAsSSA(
    TypeContext &typeCtx, mlir::ConversionPatternRewriter &rewriter,
    SmallVector<std::pair<StringRef, Value>> &vars,
    mlir::Location location) const {

  SmallVector<Value> constrs;
  auto typeInfo = typeCtx.types.find(this->getTypeName());
  if (typeInfo != typeCtx.types.end()) {
    for (size_t i = 0; i < typeInfo->getValue().paramAmount; i++) {
      constrs.push_back(rewriter.create<irdlssa::SSA_AnyType>(
          location, rewriter.getType<irdlssa::ConstraintType>()));
    }
  }

  irdlssa::SSA_ParametricType op = rewriter.create<irdlssa::SSA_ParametricType>(
      location, rewriter.getType<irdlssa::ConstraintType>(),
      this->getTypeName(), constrs);
  return op.getResult();
}

//===----------------------------------------------------------------------===//
// Attribute for constraint on dynamic type parameters
//===----------------------------------------------------------------------===//

mlir::Value DynTypeParamsConstraintAttr::registerAsSSA(
    TypeContext &typeCtx, mlir::ConversionPatternRewriter &rewriter,
    SmallVector<std::pair<StringRef, Value>> &vars,
    mlir::Location location) const {
  SmallVector<Value> constrs;
  for (auto constrAttr : this->getParamConstraints()) {
    constrs.push_back(
        constrAttr.cast<TypeConstraintAttrInterface>().registerAsSSA(
            typeCtx, rewriter, vars, location));
  }

  irdlssa::SSA_ParametricType op = rewriter.create<irdlssa::SSA_ParametricType>(
      location, rewriter.getType<irdlssa::ConstraintType>(),
      this->getTypeName(), constrs);
  return op.getResult();
}

//===----------------------------------------------------------------------===//
// Attribute for constraint on non-dynamic type parameters
//===----------------------------------------------------------------------===//

mlir::Value TypeParamsConstraintAttr::registerAsSSA(
    TypeContext &typeCtx, mlir::ConversionPatternRewriter &rewriter,
    SmallVector<std::pair<StringRef, Value>> &vars,
    mlir::Location location) const {
  SmallVector<Value> constrs;
  for (auto constrAttr : this->getParamConstraints()) {
    constrs.push_back(
        constrAttr.cast<TypeConstraintAttrInterface>().registerAsSSA(
            typeCtx, rewriter, vars, location));
  }

  irdlssa::SSA_ParametricType op = rewriter.create<irdlssa::SSA_ParametricType>(
      location, rewriter.getType<irdlssa::ConstraintType>(),
      this->getTypeDef()->getName(), constrs);
  return op.getResult();
}
