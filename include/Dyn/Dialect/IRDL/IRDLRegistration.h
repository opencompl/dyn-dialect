//===- IRDLRegistration.h - IRDL registration -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the registration of IRDL-defined MLIR objects.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_IRDL_IR_IRDLREGISTRATION_H
#define DYN_IRDL_IR_IRDLREGISTRATION_H

#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

namespace dyn {
class DynamicContext;
}

namespace irdl {

/// Register a dialect defined in IRDL in a MLIR context.
LogicalResult registerDialect(DialectOp dialectOp, dyn::DynamicContext *ctx);

} // namespace irdl
} // namespace mlir

#endif // DYN_IRDL_IR_IRDLREGISTRATION_H