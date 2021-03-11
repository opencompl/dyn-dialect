//===- DynamicContext.cpp - Dynamic context ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the creation of dynamic MLIR objects.
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "Dyn/DynamicOperation.h"
#include "Dyn/DynamicTrait.h"
#include "Dyn/DynamicType.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace dyn;

DynamicContext::DynamicContext(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<DynamicContext>()),
      ctx{ctx} {}

mlir::FailureOr<DynamicDialect *>
DynamicContext::createAndRegisterDialect(llvm::StringRef name) {
  // Allocate a new ID for the dialect.
  auto id = getTypeIDAllocator().allocateID();

  // Dialect allocator.
  auto registryCtor = [name(name.str()), this, id](MLIRContext *ctx) {
    return ctx->getOrLoadDialect(name, id, [&name, this, id]() {
      return std::make_unique<DynamicDialect>(name, this, id);
    });
  };

  DialectRegistry registry;
  registry.insert(id, name, registryCtor);

  getMLIRCtx()->appendDialectRegistry(registry);

  Dialect *dialect = getMLIRCtx()->getOrLoadDialect(name);

  // TODO, if the dialect registration failed, deallocate the TypeID.
  if (!dialect)
    return failure();

  // llvm::cast cannot be used here, since we have a custom TypeID that does
  // not correspond to the TypeID statically assigned to the DynamicDialect
  // class.
  auto *dynDialect = reinterpret_cast<DynamicDialect *>(dialect);

  dialects.insert({name, dynDialect});

  return dynDialect;
}

mlir::FailureOr<DynamicOpTrait *>
DynamicContext::registerOpTrait(llvm::StringRef name,
                                std::unique_ptr<DynamicOpTrait> opTrait) {
  auto opTraitPtr = opTrait.get();

  auto inserted = opTraits.try_emplace(name, std::move(opTrait));
  if (!inserted.second)
    return failure();

  typeIDToOpTraits.insert({opTraitPtr->getRuntimeTypeID(), opTraitPtr});

  return opTraitPtr;
}

mlir::FailureOr<DynamicOpInterface *> DynamicContext::registerOpInterface(
    llvm::StringRef name, std::unique_ptr<DynamicOpInterface> opInterface) {
  auto opInterfacePtr = opInterface.get();

  auto inserted = opInterfaces.try_emplace(name, std::move(opInterface));
  if (!inserted.second)
    return failure();

  typeIDToOpInterfaces.insert(
      {opInterfacePtr->getRuntimeTypeID(), opInterfacePtr});

  return opInterfacePtr;
}
