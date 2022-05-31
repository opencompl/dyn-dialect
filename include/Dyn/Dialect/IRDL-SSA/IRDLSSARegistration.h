//===- IRDLRegistration.h - IRDL registration -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the registration of MLIR objects from IRDL operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_IRDLSSA_IR_IRDLSSAREGISTRATION_H
#define DYN_IRDLSSA_IR_IRDLSSAREGISTRATION_H

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace irdlssa {

/// Register all the dialects in a module.
LogicalResult registerDialects(ModuleOp op);

} // namespace irdlssa
} // namespace mlir

#endif // DYN_IRDLSSA_IR_IRDLSSAREGISTRATION_H
