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

#ifndef DYN_IRDL_IR_IRDLREGISTRATION_H
#define DYN_IRDL_IR_IRDLREGISTRATION_H

#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace irdl {

/// Register a new dynamic operation in a dynamic dialect.
void registerOperation(ExtensibleDialect *dialect, StringRef name, OpDef opDef);

/// Register all the dialects in a module.
void registerDialects(ModuleOp op);

} // namespace irdl
} // namespace mlir

#endif // DYN_IRDL_IR_IRDLREGISTRATION_H
