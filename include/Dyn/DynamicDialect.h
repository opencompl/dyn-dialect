//===- DynamicDialect.h - Dynamic dialect -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Allows the creation of new dialects with runtime information.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICDIALECT_H
#define DYN_DYNAMICDIALECT_H

#include "Dyn/DynamicObject.h"
#include "Dyn/DynamicOperation.h"
#include "Dyn/DynamicType.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {
namespace dyn {

// Forward declaration.
class DynamicContext;

/// Each instance of DynamicDialect correspond to a different dialect.
class DynamicDialect : public DynamicObject, public mlir::Dialect {
public:
  /// Create a new dialect given a name.
  /// The dialect will contain no operations or types.
  DynamicDialect(llvm::StringRef name, DynamicContext *ctx);

  /// Create a new dialect given a name and an already allocated TypeID.
  /// The dialect will contain no operations or types.
  DynamicDialect(llvm::StringRef name, DynamicContext *ctx, TypeID id);

  mlir::StringRef getName() const { return name; }

  /// Create and register a new operation to the dialect.
  /// The name of the operation should not begin with the name of the dialect.
  FailureOr<DynamicOperation *> createAndAddOperation(
      llvm::StringRef name,
      std::vector<std::function<LogicalResult(Operation *)>> verifiers = {});

  /// Create and add a new type to the dialect.
  /// The name of the type should not begin with the name of the dialect.
  FailureOr<DynamicTypeDefinition *> createAndAddType(StringRef name);

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;

  /// The pointer is guaranteed to be non-null.
  /// The name format should be 'type' and not 'dialect.type'.
  FailureOr<DynamicTypeDefinition *> lookupType(StringRef name) const {
    auto it = dynTypes.find(name);
    if (it == dynTypes.end())
      return failure();
    return it->second.get();
  }

  /// The pointer is guaranteed to be non-null.
  FailureOr<DynamicTypeDefinition *> lookupType(TypeID id) const {
    auto it = typeIDToDynTypes.find(id);
    if (it == typeIDToDynTypes.end())
      return failure();
    return &*it->second;
  }

  /// The pointer is guaranteed to be non-null.
  /// The name format should be 'dialect.operation'.
  FailureOr<DynamicOperation *> lookupOp(StringRef name) const {
    auto it = dynOps.find(name);
    if (it == dynOps.end())
      return failure();
    return it->second.get();
  }

  /// The pointer is guaranteed to be non-null.
  FailureOr<DynamicOperation *> lookupOp(TypeID id) const {
    auto it = typeIDToDynOps.find(id);
    if (it == typeIDToDynOps.end())
      return failure();
    return &*it->second;
  }

private:
  /// Name of the dialect.
  /// This name is used for parsing and printing.
  const std::string name;

  /// Dynamic types registered in this dialect.
  /// Their name is stored with the format `type` and not `dialect.type`.
  llvm::StringMap<std::unique_ptr<DynamicTypeDefinition>> dynTypes{};

  /// This structure allows to get in O(1) a dynamic type given its typeID.
  /// This is useful for accessing the printer efficiently for instance.
  llvm::DenseMap<TypeID, DynamicTypeDefinition *> typeIDToDynTypes{};

  /// Dynamic operations registered in this dialect.
  /// Their name is stored with the format `op` and not `dialect.op`.
  llvm::StringMap<std::unique_ptr<DynamicOperation>> dynOps{};

  /// This structure allows to get in O(1) a dynamic type given its typeID.
  /// This is useful for accessing the verifier efficiently for instance.
  llvm::DenseMap<TypeID, DynamicOperation *> typeIDToDynOps{};

  /// Context in which the dialect is registered.
  DynamicContext *ctx;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICDIALECT_H
