//===- DynamicDialect.h - Dynamic dialect -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICDIALECT_H
#define DYN_DYNAMICDIALECT_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {
namespace dyn {

/// Forward declaration
class DynamicOperation;

class DynamicDialect {
public:
  mlir::StringRef getName() const { return name; }

  /// Create and register a dynamic operation for the dialect.
  /// Return an error if the operation could not be inserted, or if an operation
  /// with the same name already exists.
  mlir::FailureOr<DynamicOperation *> createAndRegisterOp(llvm::StringRef name);

private:
  /// Name of the dialect.
  /// This name is used for parsing and printing
  const std::string name;

  /// The dynamic operations defined by the dialect
  mlir::DenseMap<mlir::StringRef, std::unique_ptr<DynamicOperation>> ops;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICDIALECT_H
