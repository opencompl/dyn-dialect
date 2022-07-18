//===- IRDLEval.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEval.h"

using namespace mlir;
using namespace irdleval;

#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEval.cpp.inc"

#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEvalDialect.cpp.inc"

void IRDLEvalDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEvalOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEvalTypesGen.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEvalTypesGen.cpp.inc"
