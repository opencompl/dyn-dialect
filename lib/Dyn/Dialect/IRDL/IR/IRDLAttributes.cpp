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
#include "Dyn/Dialect/IRDL/TypeConstraint.h"

using namespace mlir;
using namespace dyn;
using namespace irdl;

namespace mlir {
namespace irdl {
namespace detail {

/// An attribute representing a string value.
/// This implementation already exists in MLIR, but is not public.
struct StringAttributeStorage : public AttributeStorage {
  using KeyTy = StringRef;

  StringAttributeStorage(StringRef value) : value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == KeyTy(value); }

  /// Construct a new storage instance.
  static StringAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                           const KeyTy &key) {
    return new (allocator.allocate<StringAttributeStorage>())
        StringAttributeStorage(allocator.copyInto(key));
  }

  StringRef value;
};

/// An attribute representing a reference to a type.
/// This implementation already exists in MLIR, but is not public.
struct TypeAttributeStorage : public AttributeStorage {
  using KeyTy = Type;

  TypeAttributeStorage(Type value) : value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == value; }

  /// Construct a new storage instance.
  static TypeAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                         KeyTy key) {
    return new (allocator.allocate<TypeAttributeStorage>())
        TypeAttributeStorage(key);
  }

  Type value;
};

} // namespace detail
} // namespace irdl
} // namespace mlir

//===----------------------------------------------------------------------===//
// IRDL Equality type constraint attribute with dynamic types
//===----------------------------------------------------------------------===//

EqDynTypeConstraintAttr EqDynTypeConstraintAttr::get(MLIRContext &context,
                                                     StringRef typeName) {
  return Base::get(&context, typeName);
}

FailureOr<std::unique_ptr<TypeConstraint>>
EqDynTypeConstraintAttr::getTypeConstraint(OperationOp op,
                                           DynamicContext &ctx) {
  auto constraint = EqTypeConstraint::get(getValue(), op, ctx);
  if (failed(constraint))
    return failure();

  return static_cast<std::unique_ptr<TypeConstraint>>(
      std::make_unique<EqTypeConstraint>(*constraint));
}

StringRef EqDynTypeConstraintAttr::getValue() { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// IRDL Equality type constraint attribute
//===----------------------------------------------------------------------===//

EqTypeConstraintAttr EqTypeConstraintAttr::get(MLIRContext &context,
                                               Type type) {
  return Base::get(&context, type);
}

FailureOr<std::unique_ptr<TypeConstraint>>
EqTypeConstraintAttr::getTypeConstraint(OperationOp op, DynamicContext &ctx) {
  return static_cast<std::unique_ptr<TypeConstraint>>(
      std::make_unique<EqTypeConstraint>(getValue()));
}

Type EqTypeConstraintAttr::getValue() { return getImpl()->value; }
