//===- DynamicContext.h - Dynamic context -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICCONTEXT_H
#define DYN_DYNAMICCONTEXT_H

#include "DynamicID.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace dyn {

/// Forward declaration
class DynamicDialect;

/// Forward declaration
class DynamicOperation;

/// Manages the creation and lifetime of dynamic MLIR objects such as dialects,
/// operations, types, and traits
class DynamicContext {
public:
  DynamicIDAllocator *getDynamicIDAllocator() { return &dynamicIDAllocator; }

  /// Create and register a dynamic dialect.
  /// Return an error if the dialect could not be inserted, or if a dialect with
  /// the same name was already registered.
  mlir::FailureOr<DynamicDialect *>
  createAndRegisterDialect(llvm::StringRef name);

private:
  DynamicIDAllocator dynamicIDAllocator;

  /// The set of dynamically defined dialects.
  llvm::StringMap<std::unique_ptr<DynamicDialect>> dialects;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICCONTEXT_H
