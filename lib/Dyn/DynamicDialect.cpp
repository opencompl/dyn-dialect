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

LogicalResult DynamicDialect::createAndAddTypeAlias(StringRef name, Type type) {
  auto registered = typeAliases.try_emplace(name, type);
  return success(registered.second);
}

FailureOr<DynamicOperation *> DynamicDialect::createAndAddOperation(
    StringRef name, std::vector<DynamicOperation::VerifierFn> verifiers,
    std::vector<DynamicOpTrait *> traits,
    std::vector<std::unique_ptr<DynamicOpInterfaceImpl>> interfaces) {

  // Create the interfaceMap that will contain the implementation of the
  // interfaces for this operation. Note that the actual implementation is
  // stored inside the 'DynamicOp', and the functions given to the
  // InterfaceMap will just be a redirection to the actual implementation.
  std::vector<std::pair<TypeID, void *>> interfaceMapElements;
  for (auto &interfaceImpl : interfaces) {
    auto interface = interfaceImpl->getInterface();
    interfaceMapElements.push_back(
        {interface->getRuntimeTypeID(), interface->getConcept()});
  }
  auto interfaceMap = mlir::detail::InterfaceMap(
      MutableArrayRef<std::pair<TypeID, void *>>(interfaceMapElements));

  // Register the operation to the dynamic dialect.
  auto registered = dynOps.try_emplace(
      name, new DynamicOperation(name, this, std::move(verifiers),
                                 std::move(traits), std::move(interfaces)));
  if (!registered.second)
    return failure();

  DynamicOperation *absOp = registered.first->second.get();
  auto typeID = absOp->getRuntimeTypeID();
  typeIDToDynOps.insert({typeID, absOp});
  ctx->typeIDToDynOps.insert({typeID, absOp});

  // The hasTrait implementation for this operation.
  auto hasTraitFn = [absOp](TypeID traitId) {
    return absOp->hasTrait(traitId);
  };

  AbstractOperation::insert(
      absOp->getName(), *this, absOp->getRuntimeTypeID(),
      DynamicOperation::parseOperation, DynamicOperation::printOperation,
      DynamicOperation::verifyInvariants, DynamicOperation::foldHook,
      DynamicOperation::getCanonicalizationPatterns, std::move(interfaceMap),
      hasTraitFn);

  return absOp;
}

Type DynamicDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef name;

  if (parser.parseKeyword(&name)) {
    parser.emitError(
        typeLoc,
        "dynamic dialect types should be of the format \"dialect.name\"");
    return Type();
  }

  auto type = lookupType(name);
  if (succeeded(type))
    return DynamicType::get(ctx->getMLIRCtx(), *type);

  auto alias = lookupTypeAlias(name);
  if (succeeded(alias))
    return *alias;

  parser.emitError(typeLoc, "dynamic type '")
      << name << "' was not registered in the dialect " << getName();
  return Type();
}

void DynamicDialect::printType(Type type, DialectAsmPrinter &printer) const {
  auto dynType = lookupType(type.getTypeID());
  assert(!failed(dynType));
  printer << (*dynType)->name;
}
