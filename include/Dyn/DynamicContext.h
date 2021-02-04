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

#include "Dyn/DynamicType.h"
#include "TypeIDAllocator.h"
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
class DynamicContext {
public:
  DynamicContext(MLIRContext *ctx);

  TypeIDAllocator &getTypeIDAllocator() { return typeIDAllocator; }

  MLIRContext *getMLIRCtx() { return ctx; }

  /// Create and register a dynamic dialect.
  /// Return an error if the dialect could not be inserted, or if a dialect with
  /// the same name was already registered.
  mlir::FailureOr<DynamicDialect *>
  createAndRegisterDialect(llvm::StringRef name);

  /// Get a type given its typeID.
  /// The pointer is guaranteed to be non-null.
  FailureOr<DynamicTypeDefinition *> lookupType(TypeID id) const {
    auto it = typeIDToDynTypes.find(id);
    if (it == typeIDToDynTypes.end())
      return failure();
    return &*it->second;
  }

  /// Get an operation given its typeID.
  /// The pointer is guaranteed to be non-null.
  FailureOr<DynamicOperation *> lookupOp(TypeID id) const {
    auto it = typeIDToDynOps.find(id);
    if (it == typeIDToDynOps.end())
      return failure();
    return &*it->second;
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
  llvm::DenseMap<TypeID, DynamicTypeDefinition *> typeIDToDynTypes{};

  /// This structure allows to get in O(1) a dynamic type given its typeID.
  llvm::DenseMap<TypeID, DynamicOperation *> typeIDToDynOps{};

  /// The MLIR context. It is used to register dialects, operations, types, ...
  MLIRContext *ctx;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICCONTEXT_H
