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

mlir::LogicalResult DynamicOperation::verifyInvariants(Operation *op) {
  auto typeID = op->getAbstractOperation()->typeID;

  /// This is necessary, since it is not possible to know if a dialect is
  /// dynamic or not without the DynamicContext.
  /// However, this reinterpret_cast is safe, since a DynamicOperation can
  /// only be created with a DynamicDialect.
  auto *dialect = reinterpret_cast<DynamicDialect *>(op->getDialect());
  auto dynOp = dialect->lookupOp(typeID);
  assert(!failed(dynOp));

  /// Call each custom verifier provided to the operation.
  return success(llvm::all_of((*dynOp)->verifiers, [op](auto verifier) {
    return succeeded(verifier(op));
  }));
}

DynamicOperation::DynamicOperation(
    StringRef name, DynamicDialect *dialect,
    std::vector<std::function<mlir::LogicalResult(mlir::Operation *op)>>
        verifiers)
    : DynamicObject(dialect->getDynamicContext()),
      name((dialect->getName() + "." + name).str()), dialect(dialect),
      verifiers(std::move(verifiers)) {}
