//===- IRDLEval.h - IRDL-Eval Definition ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines IRDL-Eval IR constructs.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_EVAL_IR_IRDL_EVAL_H_
#define DYN_DIALECT_IRDL_EVAL_IR_IRDL_EVAL_H_

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEvalDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEvalTypesGen.h.inc"

#define GET_OP_CLASSES
#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEvalOps.h.inc"

#endif // DYN_DIALECT_IRDL_EVAL_IR_IRDL_EVAL_H_
