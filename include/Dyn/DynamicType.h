//===- DynamicType.h - Dynamic type -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICTYPE_H
#define DYN_DYNAMICTYPE_H

#include "Dyn/DynamicContext.h"
#include "DynamicObject.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace dyn {

/// This is the definition of a dynamic type. It stores the parser and printer.
/// Each dynamic type instance refer to one instance of this class.
class DynamicTypeDefinition : public DynamicObject {
public:
  DynamicTypeDefinition(DynamicDialect *dialect, llvm::StringRef name);

  /// Dialect in which this type is defined.
  const DynamicDialect *dialect;

  /// Name of the type.
  /// Does not contain the name of the dialect beforehand.
  const std::string name;
};

/// Storage of DynamicType.
/// Contains a pointer to the type definition.
struct DynamicTypeStorage : public TypeStorage {

  using KeyTy = DynamicTypeDefinition *;

  explicit DynamicTypeStorage(DynamicTypeDefinition *typeDef)
      : typeDef(typeDef) {}

  bool operator==(const KeyTy &key) const { return typeDef == key; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static DynamicTypeStorage *construct(TypeStorageAllocator &alloc,
                                       const KeyTy &key) {
    return new (alloc.allocate<DynamicTypeStorage>()) DynamicTypeStorage(key);
  }

  /// Definition of the type.
  DynamicTypeDefinition *typeDef;
};

/// A type defined at runtime.
/// Each DynamicType instance represent a different dynamic type.
class DynamicType
    : public Type::TypeBase<DynamicType, Type, DynamicTypeStorage> {
public:
  /// Inherit Base constructors.
  using Base::Base;

  /// Get an instance of a dynamic type given a dynamic type definition.
  /// The dynamic type definition should have been registered before calling
  /// this function.
  static DynamicType get(MLIRContext *ctx, DynamicTypeDefinition *typeDef);

  /// Get the type definition of this type.
  DynamicTypeDefinition *getTypeDef();
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICTYPE_H
