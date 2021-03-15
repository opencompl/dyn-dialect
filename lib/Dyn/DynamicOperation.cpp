//===- DynamicOperation.cpp - dynamic ops -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represent operations that can be defined at runtime.
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynamicOperation.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "Dyn/DynamicInterface.h"
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
  auto *dynCtx = op->getContext()->getLoadedDialect<DynamicContext>();

  auto dynOp = dynCtx->lookupOp(typeID);
  assert(!failed(dynOp) &&
         "Trying to verify the invariants of a dynamic operation that wasn't "
         "registered in the dynamic context.");

  /// Call each custom verifier provided to the operation.
  return success(llvm::all_of((*dynOp)->verifiers, [op](auto &verifier) {
    return succeeded(verifier(op));
  }));
}

DynamicOperation::DynamicOperation(
    StringRef name, Dialect *dialect, DynamicContext *ctx,
    std::vector<VerifierFn> customVerifiers,
    std::vector<DynamicOpTrait *> traits,
    std::vector<std::unique_ptr<DynamicOpInterfaceImpl>> interfacesImpl)
    : DynamicObject(ctx), name((dialect->getNamespace() + "." + name).str()),
      dialect(dialect), verifiers(std::move(customVerifiers)) {
  // Add traits to verifiers and traitIDs.
  for (auto trait : traits) {
    traitsId.push_back(trait->getRuntimeTypeID());
    verifiers.push_back(
        [trait](Operation *op) { return trait->verifyTrait(op); });
  }

  for (auto &interface : interfacesImpl)
    interfaces.push_back(
        {interface->getInterface()->getRuntimeTypeID(), std::move(interface)});
}

bool DynamicOperation::hasTrait(TypeID traitId) {
  return llvm::any_of(traitsId, [traitId](auto id) { return id == traitId; });
}

DynamicOpInterfaceImpl *
DynamicOperation::getInterfaceImpl(DynamicOpInterface *interface) {
  auto res = llvm::find_if(interfaces, [interface](const auto &p) {
    return p.first == interface->getRuntimeTypeID();
  });

  assert(res != interfaces.end() &&
         "Trying to get an interface implementation in an operation that "
         "doesn't implement the interface");
  return res->second.get();
}
