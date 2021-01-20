//===- DynamicOperation.h - Dynamic operation -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICOPERATION_H
#define DYN_DYNAMICOPERATION_H

#include "DynamicDialect.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace dyn {

/// Forward declaration
class DynamicDialect;

/// Each instance of DynamicOperation correspond to a different operation
class DynamicOperation {
public:
  /// Create a new dynamic operation given the operation name and the defining
  /// dialect.
  /// The operation name should be `operation` and not `dialect.operation`.
  DynamicOperation(mlir::StringRef name, DynamicDialect *dialect);

private:
  /// Full name of the operation: `dialect.operation`.
  std::string name;

  /// Pointer to the dialect defining the operation.
  DynamicDialect *dialect;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICOPERATION_H
