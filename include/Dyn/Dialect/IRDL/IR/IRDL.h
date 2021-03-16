//===- IRDL.h - IR Definition Language dialect ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the dialect for the IR Definition Language.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_IRDL_H_
#define DYN_DIALECT_IRDL_IR_IRDL_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include <memory>

// Forward declaration.
namespace mlir {
namespace irdl {
class OpTypeDef;
class OpTypeDefAttr;
} // namespace irdl

namespace dyn {
class DynamicDialect;
} // namespace dyn
} // namespace mlir

//===----------------------------------------------------------------------===//
// IRDL Dialect
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDLOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// IRDL Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dyn/Dialect/IRDL/IR/IRDLOps.h.inc"

#endif // DYN_DIALECT_IRDL_IR_IRDL_H_
