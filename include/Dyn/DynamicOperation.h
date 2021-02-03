//===- DynamicOperation.h - Dynamic operation -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represent operations that can be defined at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICOPERATION_H
#define DYN_DYNAMICOPERATION_H

#include "Dyn/DynamicObject.h"
#include "DynamicDialect.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <mlir/IR/OpDefinition.h>

namespace mlir {
namespace dyn {

// Forward declaration.
class DynamicDialect;

/// Each instance of DynamicOperation correspond to a different operation.
class DynamicOperation : public mlir::Op<DynamicOperation>,
                         public DynamicObject {
public:
  /// Create a new dynamic operation given the operation name and the defining
  /// dialect.
  /// The operation name should be `operation` and not `dialect.operation`.
  DynamicOperation(
      mlir::StringRef name, DynamicDialect *dialect,
      std::vector<std::function<mlir::LogicalResult(mlir::Operation *op)>>
          verifiers = {});

  /// Get the operation name.
  /// The name should have the format `dialect.name`.
  StringRef getName() { return name; }

  /// Parse a dynamic operation.
  static mlir::ParseResult parseOperation(OpAsmParser &parser,
                                          OperationState &result) {
    return success();
  }

  /// Print a dynamic operation.
  static void printOperation(Operation *op, OpAsmPrinter &printer);

  /// Verify invariants of a dynamic operation.
  static mlir::LogicalResult verifyInvariants(Operation *op);

  /// Fold hook of generic operations.
  static mlir::LogicalResult
  foldHook(mlir::Operation *op, llvm::ArrayRef<mlir::Attribute> operands,
           llvm::SmallVectorImpl<mlir::OpFoldResult> &results) {
    return failure();
  }

  /// Check if the operation has a specific trait given the trait TypeID.
  static bool hasTrait(TypeID traitId) { return false; }

private:
  /// Full name of the operation: `dialect.operation`.
  const std::string name;

  /// Pointer to the dialect defining the operation.
  DynamicDialect *dialect;

  /// Custom verifiers for the operation.
  std::vector<std::function<mlir::LogicalResult(mlir::Operation *op)>>
      verifiers;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICOPERATION_H
