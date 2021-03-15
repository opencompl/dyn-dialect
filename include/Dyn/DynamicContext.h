//===- DynamicContext.h - Dynamic context -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the creation of dynamic MLIR objects.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICCONTEXT_H
#define DYN_DYNAMICCONTEXT_H

#include "Dyn/DynamicInterface.h"
#include "Dyn/DynamicTrait.h"
#include "Dyn/DynamicType.h"
#include "TypeIDAllocator.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {

// Forward declaration.
class MLIRContext;

namespace dyn {

// Forward declaration.
class DynamicTypeDefinition;

// Forward declaration.
class DynamicDialect;

// Forward declaration.
class DynamicOperation;

/// Manages the creation and lifetime of dynamic MLIR objects such as dialects,
/// operations, types, and traits
/// The dynamic context is a dialect so we can get the instance through
/// `MLIRContext::getLoadedDialect`. This is a bit of a hack though.
class DynamicContext : public mlir::Dialect {
public:
  static StringRef getDialectNamespace() { return "DynamicContext"; }

  explicit DynamicContext(mlir::MLIRContext *ctx);

  TypeIDAllocator &getTypeIDAllocator() { return typeIDAllocator; }

  MLIRContext *getMLIRCtx() { return ctx; }

  /// Create and register a dynamic dialect.
  /// Return an error if the dialect could not be inserted, or if a dialect with
  /// the same name was already registered.
  mlir::FailureOr<DynamicDialect *>
  createAndRegisterDialect(llvm::StringRef name);

  /// Create and add a new dynamic operation to an existing dialect.
  /// Its name should be in the format 'opname' and not 'dialectname.opname'.
  mlir::FailureOr<DynamicOperation *> createAndRegisterOperation(
      StringRef name, Dialect *dialect,
      std::vector<
          llvm::unique_function<mlir::LogicalResult(mlir::Operation *op)>>
          verifiers,
      std::vector<DynamicOpTrait *> traits,
      std::vector<std::unique_ptr<DynamicOpInterfaceImpl>> interfaces);

  /// Create and add a new type to the dialect.
  /// The name of the type should not begin with the name of the dialect.
  mlir::FailureOr<DynamicTypeDefinition *>
  createAndRegisterType(StringRef name, Dialect *dialect);

  /// Create and add a new type alias to the dialect.
  /// The name of the type alias should not begin with the name of the dialect.
  LogicalResult addTypeAlias(StringRef name, Dialect *dialect, Type type);

private:
  /// Register a dynamic trait.
  /// Return an error if a trait with the same name was already registered.
  mlir::FailureOr<DynamicOpTrait *>
  registerOpTrait(llvm::StringRef name,
                  std::unique_ptr<DynamicOpTrait> opTrait);

  /// Register a dynamic interface.
  /// Return an error if an interface with the same name was already registered.
  mlir::FailureOr<DynamicOpInterface *>
  registerOpInterface(llvm::StringRef name,
                      std::unique_ptr<DynamicOpInterface> opInterface);

public:
  /// Create and register a dynamic trait.
  /// Return an error if a trait with the same name was already registered.
  mlir::FailureOr<DynamicOpTrait *>
  createAndRegisterOpTrait(llvm::StringRef name, OpTraitVerifierFn verifier) {
    return registerOpTrait(name, std::make_unique<DynamicOpTrait>(
                                     this, name, std::move(verifier)));
  }

  /// Create and register a wrapper around a c++-defined trait.
  /// Return an error if a trait with the same name was already registered.
  template <template <typename ConcreteT> class TraitTy>
  mlir::FailureOr<DynamicOpTrait *>
  createAndRegisterOpTrait(llvm::StringRef name) {
    return registerOpTrait(name,
                           std::move(DynamicOpTrait::get<TraitTy>(this, name)));
  }

  /// Create and register a dynamic interface defined in C++.
  /// Return an error if a trait with the same name was already registered.
  template <typename InterfaceTy>
  mlir::FailureOr<DynamicOpInterface *>
  createAndRegisterOpInterface(llvm::StringRef name) {
    return registerOpInterface(
        name, std::unique_ptr<InterfaceTy>(new InterfaceTy(this)));
  }

  /// Get an operation trait given its name.
  /// The pointer is guaranteed to be non-null.
  FailureOr<DynamicOpTrait *> lookupOpTrait(StringRef name) const {
    auto it = opTraits.find(name);
    if (it == opTraits.end())
      return failure();
    return &*it->second;
  }

  /// Get an operation trait given its typeID.
  /// The pointer is guaranteed to be non-null.
  FailureOr<DynamicOpTrait *> lookupOpTrait(TypeID id) const {
    auto it = typeIDToOpTraits.find(id);
    if (it == typeIDToOpTraits.end())
      return failure();
    return &*it->second;
  }

  /// Get an operation interface given its name.
  /// The pointer is guaranteed to be non-null.
  FailureOr<DynamicOpInterface *> lookupOpInterface(StringRef name) const {
    auto it = opInterfaces.find(name);
    if (it == opInterfaces.end())
      return failure();
    return &*it->second;
  }

  /// Get an operation interface given its typeID.
  /// The pointer is guaranteed to be non-null.
  FailureOr<DynamicOpInterface *> lookupOpInterface(TypeID id) const {
    auto it = typeIDToOpInterfaces.find(id);
    if (it == typeIDToOpInterfaces.end())
      return failure();
    return &*it->second;
  }

  /// Get a dialect given its typeID.
  /// The pointer is guaranteed to be non-null.
  FailureOr<DynamicDialect *> lookupDialect(StringRef name) const {
    auto it = dialects.find(name);
    if (it == dialects.end())
      return failure();
    return &*it->second;
  }

  /// The pointer is guaranteed to be non-null.
  /// The name format should be 'dialect.operation'.
  FailureOr<DynamicOperation *> lookupOp(StringRef name) const {
    auto it = nameToDynOps.find(name);
    if (it == nameToDynOps.end())
      return failure();
    return &*it->second;
  }

  /// The pointer is guaranteed to be non-null.
  FailureOr<DynamicOperation *> lookupOp(TypeID id) const {
    auto it = dynOps.find(id);
    if (it == dynOps.end())
      return failure();
    return it->second.get();
  }

  /// The pointer is guaranteed to be non-null.
  /// The name format should be 'type' and not 'dialect.type'.
  FailureOr<DynamicTypeDefinition *> lookupType(StringRef name) const {
    auto it = nameToDynTypes.find(name);
    if (it == nameToDynTypes.end())
      return failure();
    return &*it->second;
  }

  /// The pointer is guaranteed to be non-null.
  FailureOr<DynamicTypeDefinition *> lookupType(TypeID id) const {
    auto it = dynTypes.find(id);
    if (it == dynTypes.end())
      return failure();
    return it->second.get();
  }

  /// The name format should be 'dialectname.aliasname'.
  FailureOr<Type> lookupTypeAlias(StringRef name) const {
    auto it = typeAliases.find(name);
    if (it == typeAliases.end())
      return failure();
    return Type(it->second);
  }

  /// The name format should be 'dialectname.typename'.
  FailureOr<Type> lookupTypeOrTypeAlias(StringRef name) const {
    auto dynType = lookupType(name);
    if (succeeded(dynType))
      return DynamicType::get(getContext(), *dynType);

    return lookupTypeAlias(name);
  }

  /// We declare DynamicDialect friend so it can register types and operations
  /// in the context.
  friend DynamicDialect;

private:
  /// TypeID allocator used for dialects, operations, types, ...
  TypeIDAllocator typeIDAllocator;

  /// The set of dynamically defined dialects.
  llvm::StringMap<DynamicDialect *> dialects;

  /// This structure allows to get in O(1) a dynamic type given its typeID.
  llvm::DenseMap<TypeID, DynamicTypeDefinition *> typeIDToDynTypes;

  /// The set of dynamically defined operation traits.
  llvm::StringMap<std::unique_ptr<DynamicOpTrait>> opTraits;

  /// This structure allows to get in O(1) a dynamic trait given its typeID.
  llvm::DenseMap<TypeID, DynamicOpTrait *> typeIDToOpTraits;

  /// The set of dynamically defined operation traits.
  llvm::StringMap<std::unique_ptr<DynamicOpInterface>> opInterfaces;

  /// This structure allows to get in O(1) a dynamic trait given its typeID.
  llvm::DenseMap<TypeID, DynamicOpInterface *> typeIDToOpInterfaces;

  /// The set of all dynamic operations registered.
  llvm::DenseMap<TypeID, std::unique_ptr<DynamicOperation>> dynOps;

  /// This structure allows to get in O(1) a dynamic operation given its name.
  /// The name format should be 'dialect.opname'.
  llvm::StringMap<DynamicOperation *> nameToDynOps;

  /// The set of all dynamic types registered.
  llvm::DenseMap<TypeID, std::unique_ptr<DynamicTypeDefinition>> dynTypes;

  /// This structure allows to get in O(1) a dynamic type given its name.
  /// The name format should be 'dialect.type'.
  llvm::StringMap<DynamicTypeDefinition *> nameToDynTypes;

  /// Type aliases registered in this dialect.
  /// Their name is stored with the format `alias` and not `dialect.alias`.
  llvm::StringMap<Type> typeAliases;

  /// The MLIR context. It is used to register dialects, operations, types, ...
  MLIRContext *ctx;

public:
  /// This field is used during parsing, and may be needed to be moved somewhere
  /// else. If this field is non-null, it points to the dialect that is
  /// currently being parsed by MLIR.
  DynamicDialect *currentlyParsedDialect = nullptr;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICCONTEXT_H
