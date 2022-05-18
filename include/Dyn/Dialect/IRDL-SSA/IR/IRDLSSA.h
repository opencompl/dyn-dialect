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

#ifndef DYN_DIALECT_IRDL_SSA_IR_IRDL_SSA_H_
#define DYN_DIALECT_IRDL_SSA_IR_IRDL_SSA_H_

#include "Dyn/Dialect/IRDL/IR/IRDLTraits.h"
#include "Dyn/Dialect/IRDL/TypeWrapper.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include <memory>

// Forward declaration.
namespace mlir {
namespace irdlssa {
class OpDef;
class OpDefAttr;
} // namespace irdlssa
} // namespace mlir

//===----------------------------------------------------------------------===//
// IRDL-SSA Dialect
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSADialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSATypesGen.h.inc"

#define GET_OP_CLASSES
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSAOps.h.inc"

#endif // DYN_DIALECT_IRDL_SSA_IR_IRDL_SSA_H_
