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
#include "mlir/IR/Diagnostics.h"
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
  /// Type of the verifier function.
  using VerifierFn = llvm::unique_function<LogicalResult(
      function_ref<InFlightDiagnostic()>, ArrayRef<Attribute>) const>;

  DynamicTypeDefinition(Dialect *dialect, llvm::StringRef name,
                        VerifierFn verifier);

  /// Dialect in which this type is defined.
  const Dialect *dialect;

  /// Name of the type.
  /// Does not contain the name of the dialect beforehand.
  const std::string name;

  /// Check that the type parameters are valid.
  LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<Attribute> params) const {
    return verifier(emitError, params);
  }

private:
  /// Verifier for the type parameters.
  VerifierFn verifier;
};

/// Storage of DynamicType.
/// Contains a pointer to the type definition.
struct DynamicTypeStorage : public TypeStorage {

  using KeyTy = std::pair<DynamicTypeDefinition *, ArrayRef<Attribute>>;

  explicit DynamicTypeStorage(DynamicTypeDefinition *typeDef,
                              ArrayRef<Attribute> params)
      : typeDef(typeDef), params(params) {}

  bool operator==(const KeyTy &key) const {
    return typeDef == key.first && params == key.second;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static DynamicTypeStorage *construct(TypeStorageAllocator &alloc,
                                       const KeyTy &key) {
    return new (alloc.allocate<DynamicTypeStorage>())
        DynamicTypeStorage(key.first, alloc.copyInto(key.second));
  }

  /// Definition of the type.
  DynamicTypeDefinition *typeDef;

  /// The type parameters.
  ArrayRef<Attribute> params;
};

/// A type defined at runtime.
/// Each DynamicType instance represent a different dynamic type.
class DynamicType
    : public Type::TypeBase<DynamicType, Type, DynamicTypeStorage> {
public:
  /// Inherit Base constructors.
  using Base::Base;

  /// Get an instance of a dynamic type given a dynamic type definition and
  /// type parameters.
  /// The dynamic type definition should have been registered before calling
  /// this function.
  static DynamicType get(DynamicTypeDefinition *typeDef,
                         ArrayRef<Attribute> params = {});

  /// Get an instance of a dynamic type given a dynamic type definition and type
  /// parameter. Also check if the parameters are valid.
  /// The dynamic type definition should have been registered before calling
  /// this function.
  static DynamicType getChecked(function_ref<InFlightDiagnostic()> emitError,
                                DynamicTypeDefinition *typeDef,
                                ArrayRef<Attribute> params = {});

  /// Get the type definition of this type.
  DynamicTypeDefinition *getTypeDef();

  /// Check if a type is a specific dynamic type.
  static bool isa(Type type, DynamicTypeDefinition *typeDef) {
    return type.getTypeID() == typeDef->getRuntimeTypeID();
  }

  /// Check if a type is a dynamic type.
  static bool classof(Type type);

  /// Parse the dynamic type parameters and construct the type.
  /// The parameters are either empty, and nothing is parsed,
  /// or they are in the format '<>' or '<attr (,attr)*>'.
  static ParseResult parse(DialectAsmParser &parser,
                           DynamicTypeDefinition *typeDef,
                           DynamicType &parsedType);

  /// Print the dynamic type with the format
  /// 'type' or 'type<>' if there is no parameters, or 'type<attr (,attr)*>'.
  void print(DialectAsmPrinter &printer);

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
