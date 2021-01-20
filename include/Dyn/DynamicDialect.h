//===- DynamicDialect.h - Dynamic dialect -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICDIALECT_H
#define DYN_DYNAMICDIALECT_H

#include "Dyn/DynamicObject.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {
namespace dyn {

/// Forward declaration
class DynamicOperation;

/// Forward declaration
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

private:
  /// Name of the dialect.
  /// This name is used for parsing and printing
  const std::string name;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICDIALECT_H
