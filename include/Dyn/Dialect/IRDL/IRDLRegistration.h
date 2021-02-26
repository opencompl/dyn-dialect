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
#include "mlir/Support/LogicalResult.h"

namespace mlir {

// Forward declaration.
namespace dyn {
class DynamicContext;
class DynamicDialect;
} // namespace dyn

namespace irdl {

/// Register a new dynamic type in a dynamic dialect.
LogicalResult registerType(dyn::DynamicDialect *dialect, StringRef name);

/// Register a new type alias in a dynamic dialect.
LogicalResult registerTypeAlias(dyn::DynamicDialect *dialect, StringRef name,
                                Type type);

/// Register a new dynamic operation in a dynamic dialect.
LogicalResult registerOperation(dyn::DynamicDialect *dialect, StringRef name,
                                OpTypeDef opTypeDef);

} // namespace irdl
} // namespace mlir

#endif // DYN_IRDL_IR_IRDLREGISTRATION_H
