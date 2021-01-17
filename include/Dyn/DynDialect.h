//===- DynDialect.h - Dynamic dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNDIALECT_H
#define DYN_DYNDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace dyn {

class DynDialect : public mlir::Dialect {
public:
  DynDialect(mlir::MLIRContext *ctx);

  static mlir::StringRef getDialectNamespace() { return "dyn"; }
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNDIALECT_H
