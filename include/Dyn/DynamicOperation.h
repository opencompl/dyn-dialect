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
#include "Dyn/DynamicTrait.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include <mlir/IR/OpDefinition.h>

namespace mlir {
namespace dyn {

// Forward declaration.
class DynamicDialect;
class DynamicOpInterfaceImpl;
class DynamicOpInterface;

/// Each instance of DynamicOperation correspond to a different operation.
class DynamicOperation : public DynamicObject {
public:
  using VerifierFn =
      llvm::unique_function<mlir::LogicalResult(mlir::Operation *op)>;

  /// Create a new dynamic operation given the operation name and the defining
  /// dialect.
  /// The operation name should be `operation` and not `dialect.operation`.
  DynamicOperation(
      mlir::StringRef name, Dialect *dialect, DynamicContext *ctx,
      std::vector<VerifierFn> verifiers, std::vector<DynamicOpTrait *> traits,
      std::vector<std::unique_ptr<DynamicOpInterfaceImpl>> interfaces);

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

  /// Get the canonicalization patterns of this operation.
  /// There is no way to define canonicalization patterns yet.
  static void getCanonicalizationPatterns(OwningRewritePatternList &,
                                          MLIRContext *ctx) {}

  /// Check if the operation has a specific trait given the trait TypeID.
  bool hasTrait(TypeID traitId);

  DynamicOpInterfaceImpl *getInterfaceImpl(DynamicOpInterface *interface);

private:
  /// Full name of the operation: `dialect.operation`.
  const std::string name;

  /// Dialect that defines this operation.
  Dialect *dialect;

  /// Custom verifiers for the operation.
  std::vector<VerifierFn> verifiers;

  // Operation traits TypeID.
  std::vector<TypeID> traitsId;

  /// Interfaces and their implementations.
  std::vector<std::pair<TypeID, std::unique_ptr<DynamicOpInterfaceImpl>>>
      interfaces;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICOPERATION_H
