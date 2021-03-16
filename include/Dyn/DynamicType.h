//===- DynamicType.h - Dynamic type -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represent types that can be defined at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICTYPE_H
#define DYN_DYNAMICTYPE_H

#include "DynamicObject.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {

// Forward declaration.
class DialectAsmPrinter;
class DialectAsmParser;

namespace dyn {

// Forward declaration.
class DynamicDialect;

/// This is the definition of a dynamic type. It stores the parser and
/// printer. Each dynamic type instance refer to one instance of this class.
class DynamicTypeDefinition : public DynamicObject {
public:
  DynamicTypeDefinition(Dialect *dialect, llvm::StringRef name);

  /// Dialect in which this type is defined.
  const Dialect *dialect;

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

  /// Check if a type is a specific dynamic type.
  static bool isa(Type type, DynamicTypeDefinition *typeDef) {
    return type.getTypeID() == typeDef->getRuntimeTypeID();
  }

  /// Parse the dynamic type 'typeName' in the dialect 'dialect'.
  /// If there is no such dynamic type, returns no value.
  /// If there is such dynamic type, then parse it, and returns the parse
  /// result.
  /// If this succeed, put the resulting type in 'resultType'.
  static OptionalParseResult parseOptionalDynamicType(const Dialect *dialect,
                                                      StringRef typeName,
                                                      DialectAsmParser &parser,
                                                      Type &resultType);

  /// If 'type' is a dynamic type, print it.
  /// Returns success if the type was printed, and failure if the type was not a
  /// dynamic type.
  static LogicalResult printIfDynamicType(Type type,
                                          DialectAsmPrinter &printer);
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICTYPE_H
