//===- DynamicOperation.cpp - dynamic ops -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynamicOperation.h"
#include "Dyn/DynamicDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace dyn;

void DynamicOperation::printOperation(Operation *op, OpAsmPrinter &printer) {
  printer.printGenericOp(op);
}

DynamicOperation::DynamicOperation(StringRef name, DynamicDialect *dialect)
    : DynamicObject(dialect->getDynamicContext()),
    name((dialect->getName() + "." + name).str()), dialect{dialect} {}
