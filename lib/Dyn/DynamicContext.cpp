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
#include "Dyn/DynamicInterface.h"
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

  return dynDialect;
}

LogicalResult DynamicContext::createAndRegisterOperation(
    StringRef name, Dialect *dialect, AbstractOperation::ParseAssemblyFn parser,
    AbstractOperation::PrintAssemblyFn printer,
    AbstractOperation::VerifyInvariantsFn verifier,
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

  auto typeID = typeIDAllocator.allocateID();
  auto opName = (dialect->getNamespace() + "." + name).str();

  std::vector<std::pair<TypeID, std::unique_ptr<DynamicOpInterfaceImpl>>>
      interfacesImpl;

  for (auto &interfaceImpl : interfaces) {
    auto interfaceID = interfaceImpl->getInterface()->getRuntimeTypeID();
    interfacesImpl.emplace_back(interfaceID, std::move(interfaceImpl));
  }
  opInterfaceImpls.insert(std::make_pair(typeID, std::move(interfacesImpl)));

  std::vector<TypeID> traitsId = {};
  for (auto *trait : traits)
    traitsId.push_back(trait->getRuntimeTypeID());

  // The hasTrait implementation for this operation.
  auto hasTraitFn = [traitsId{std::move(traitsId)}](TypeID traitId) {
    return llvm::any_of(traitsId, [traitId](auto id) { return id == traitId; });
  };

  auto foldHook = [](mlir::Operation *op,
                     llvm::ArrayRef<mlir::Attribute> operands,
                     llvm::SmallVectorImpl<mlir::OpFoldResult> &results) {
    return failure();
  };

  auto getCanonicalizationPatterns = [](OwningRewritePatternList &,
                                        MLIRContext *) {};

  AbstractOperation::insert(opName, *dialect, typeID, std::move(parser),
                            std::move(printer), std::move(verifier),
                            std::move(foldHook),
                            std::move(getCanonicalizationPatterns),
                            std::move(interfaceMap), std::move(hasTraitFn));

  return success();
}

FailureOr<DynamicTypeDefinition *> DynamicContext::createAndRegisterType(
    StringRef name, Dialect *dialect,
    DynamicTypeDefinition::VerifierFn verifier) {
  auto fullName = (dialect->getNamespace() + "." + name).str();
  auto *dynType = new DynamicTypeDefinition(dialect, name, std::move(verifier));
  auto typeID = dynType->getRuntimeTypeID();

  // If an alias with the same name is already defined, fail.
  if (typeAliases.count(fullName))
    return failure();

  // If a type with the same name is already defined, fail.
  auto registered = dynTypes.try_emplace(typeID, dynType);
  if (!registered.second)
    return failure();

  nameToDynTypes.insert({fullName, dynType});

  /// Add the type to the dialect and the type uniquer.
  addType(typeID, AbstractType(*dialect, detail::InterfaceMap::template get<>(),
                               typeID));
  detail::TypeUniquer::registerType<DynamicType>(ctx, typeID);

  return dynType;
}

LogicalResult DynamicContext::addTypeAlias(StringRef name, Dialect *dialect,
                                           Type type) {
  auto fullName = (dialect->getNamespace() + "." + name).str();
  auto registered = typeAliases.try_emplace(fullName, type);
  return success(registered.second);
}

mlir::FailureOr<DynamicOpTrait *>
DynamicContext::registerOpTrait(std::unique_ptr<DynamicOpTrait> opTrait) {
  auto traitName = opTrait->name;
  auto opTraitPtr = opTrait.get();

  auto inserted = opTraits.try_emplace(traitName, std::move(opTrait));
  if (!inserted.second)
    return failure();

  typeIDToOpTraits.insert({opTraitPtr->getRuntimeTypeID(), opTraitPtr});

  return opTraitPtr;
}

mlir::FailureOr<DynamicOpInterface *> DynamicContext::registerOpInterface(
    std::unique_ptr<DynamicOpInterface> opInterface) {
  auto interfaceName = opInterface->name;
  auto opInterfacePtr = opInterface.get();

  auto inserted =
      opInterfaces.try_emplace(interfaceName, std::move(opInterface));
  if (!inserted.second)
    return failure();

  typeIDToOpInterfaces.insert(
      {opInterfacePtr->getRuntimeTypeID(), opInterfacePtr});

  return opInterfacePtr;
}
