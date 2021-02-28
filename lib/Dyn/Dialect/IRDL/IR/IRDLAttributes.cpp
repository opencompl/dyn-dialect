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
// IRDL Equality type constraint attribute
//===----------------------------------------------------------------------===//

EqTypeConstraintAttr EqTypeConstraintAttr::get(MLIRContext &context,
                                               Type type) {
  return Base::get(&context, type);
}

std::unique_ptr<TypeConstraint>
EqTypeConstraintAttr::getTypeConstraint(DynamicContext &ctx) {
  return std::make_unique<EqTypeConstraint>(getValue());
}

Type EqTypeConstraintAttr::getValue() { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// IRDL AnyOf type constraint attribute
//===----------------------------------------------------------------------===//

namespace mlir {
namespace irdl {
namespace detail {

/// An attribute storage containing a reference to an array of type.
struct TypeArrayAttrStorage : public AttributeStorage {
  using KeyTy = ArrayRef<Type>;

  TypeArrayAttrStorage(ArrayRef<Type> values) : values(values) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == values; }

  /// Construct a new storage instance.
  static TypeArrayAttrStorage *construct(AttributeStorageAllocator &allocator,
                                         KeyTy key) {
    return new (allocator.allocate<TypeArrayAttrStorage>())
        TypeArrayAttrStorage(allocator.copyInto(key));
  }

  ArrayRef<Type> values;
};

} // namespace detail
} // namespace irdl
} // namespace mlir

AnyOfTypeConstraintAttr AnyOfTypeConstraintAttr::get(MLIRContext &context,
                                                     ArrayRef<Type> type) {
  return Base::get(&context, type);
}

std::unique_ptr<TypeConstraint>
AnyOfTypeConstraintAttr::getTypeConstraint(DynamicContext &ctx) {
  return std::make_unique<AnyOfTypeConstraint>(getValue());
}

ArrayRef<Type> AnyOfTypeConstraintAttr::getValue() { return getImpl()->values; }

//===----------------------------------------------------------------------===//
// Always true type constraint attribute
//===----------------------------------------------------------------------===//

AnyTypeConstraintAttr AnyTypeConstraintAttr::get(MLIRContext &context) {
  return Base::get(&context);
}

std::unique_ptr<TypeConstraint>
AnyTypeConstraintAttr::getTypeConstraint(DynamicContext &ctx) {
  return std::make_unique<AnyTypeConstraint>();
}
