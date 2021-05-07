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

#include "Dyn/DynamicInterface.h"
#include "Dyn/DynamicObject.h"
#include "Dyn/DynamicTrait.h"
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
class DynamicOpTrait;
class DynamicOpInterfaceImpl;

/// Each instance of DynamicDialect correspond to a different dialect.
class DynamicDialect : public DynamicObject, public mlir::Dialect {
public:
  /// Create a new dialect given a name.
  /// The dialect will contain no operations or types.
  DynamicDialect(llvm::StringRef name, DynamicContext *ctx);

  /// Create a new dialect given a name and an already allocated TypeID.
  /// The dialect will contain no operations or types.
  DynamicDialect(llvm::StringRef name, DynamicContext *ctx, TypeID id);

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;

private:
  /// Context in which the dialect is registered.
  DynamicContext *ctx;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICDIALECT_H
