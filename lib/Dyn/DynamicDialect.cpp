//===- DynamicDialect.cpp - Dynamic dialects --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Allows the creation of new dialects with runtime information.
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynamicDialect.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicObject.h"
#include "Dyn/DynamicOperation.h"
#include "Dyn/DynamicType.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace dyn;

DynamicDialect::DynamicDialect(llvm::StringRef name, DynamicContext *ctx)
    : DynamicObject{ctx}, Dialect(name, ctx->getMLIRCtx(), getRuntimeTypeID()),
      name{name}, ctx{ctx} {}

DynamicDialect::DynamicDialect(llvm::StringRef name, DynamicContext *ctx,
                               TypeID id)
    : DynamicObject{ctx, id},
      Dialect(name, ctx->getMLIRCtx(), id), name{name}, ctx{ctx} {}

FailureOr<DynamicTypeDefinition *>
DynamicDialect::createAndAddType(StringRef name) {
  /// If a type with same name is already defined, fail.
  auto registered =
      dynTypes.try_emplace(name, new DynamicTypeDefinition(this, name));
  if (!registered.second)
    return failure();

  DynamicTypeDefinition *type = registered.first->second.get();
  auto typeID = type->getRuntimeTypeID();
  typeIDToDynTypes.insert({typeID, type});
  ctx->typeIDToDynTypes.insert({typeID, type});

  /// Add the type to the dialect and the type uniquer.
  addType(typeID,
          AbstractType(*this, detail::InterfaceMap::template get<>(), typeID));
  detail::TypeUniquer::registerType<DynamicType>(ctx->getMLIRCtx(), typeID);

  return type;
}

FailureOr<DynamicOperation *> DynamicDialect::createAndAddOperation(
    StringRef name,
    std::vector<std::function<LogicalResult(Operation *)>> verifiers) {
  auto registered = dynOps.try_emplace(
      name, new DynamicOperation(name, this, std::move(verifiers)));
  if (!registered.second)
    return failure();

  DynamicOperation *absOp = registered.first->second.get();
  auto typeID = absOp->getRuntimeTypeID();
  typeIDToDynOps.insert({typeID, absOp});
  ctx->typeIDToDynOps.insert({typeID, absOp});

  AbstractOperation::insert(
      absOp->getName(), *this, {}, absOp->getRuntimeTypeID(),
      DynamicOperation::parseOperation, DynamicOperation::printOperation,
      DynamicOperation::verifyInvariants, DynamicOperation::foldHook,
      DynamicOperation::getCanonicalizationPatterns,
      detail::InterfaceMap::template get<>(), DynamicOperation::hasTrait);

  return absOp;
}

Type DynamicDialect::parseType(mlir::DialectAsmParser &parser) const {
  StringRef name;
  if (parser.parseKeyword(&name))
    return {};

  auto type = lookupType(name);
  if (failed(type))
    return {};

  return DynamicType::get(ctx->getMLIRCtx(), *type);
}

void DynamicDialect::printType(Type type, DialectAsmPrinter &printer) const {
  auto dynType = lookupType(type.getTypeID());
  assert(!failed(dynType));
  printer << (*dynType)->name;
}
