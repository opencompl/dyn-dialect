//===- StandardInterfaces.cpp - Std interface definitions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the attributes used in the IRDL dialect to represent
// standard interfaces.
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/StandardOpInterfaces.h"
#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
#include "Dyn/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "Dyn/DynamicInterface.h"
#include "Dyn/DynamicOperation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_ATTRDEF_CLASSES
#include "Dyn/Dialect/IRDL/IR/StandardOpInterfaces.cpp.inc"

using mlir::OpAsmParser;

using namespace mlir;
using namespace irdl;

void IRDLDialect::registerStandardInterfaceAttributes() {
  addNewAttribute<irdl::DynMemoryEffectOpInterfaceAttr>();
}

namespace {

/// Get the list of available effects, and their names.
std::vector<std::pair<StringRef, MemoryEffects::Effect *>> availableEffects() {
  return {std::make_pair("Allocate", MemoryEffects::Allocate::get()),
          std::make_pair("Free", MemoryEffects::Free::get()),
          std::make_pair("Read", MemoryEffects::Read::get()),
          std::make_pair("Write", MemoryEffects::Write::get())};
};

/// Check if there exist an effect with the given name.
bool isEffectName(StringRef name) {
  return llvm::any_of(availableEffects(),
                      [name](auto a) { return a.first == name; });
}

/// Get the effect associated with the name.
MemoryEffects::Effect *getEffect(StringRef name) {
  assert(isEffectName(name) && "getEffect called with an invalid name");
  return llvm::find_if(availableEffects(),
                       [name](auto a) { return a.first == name; })
      ->second;
}

/// Get the name associated with the effect.
StringRef getEffectName(MemoryEffects::Effect *effect) {
  return llvm::find_if(availableEffects(),
                       [effect](auto a) { return a.second == effect; })
      ->first;
}

} // namespace

std::unique_ptr<dyn::DynamicOpInterfaceImpl>
DynMemoryEffectOpInterfaceAttr::getInterfaceImpl() const {
  auto effects = getEffects();
  auto dynCtx = getContext()->getLoadedDialect<dyn::DynamicContext>();

  return std::make_unique<DynMemoryEffectOpInterfaceImpl>(dynCtx,
                                                          std::move(effects));
}

namespace {
/// Parse a memory effect name.
/// Emit an error in case the name is not a memory effect name.
ParseResult parseEffectName(mlir::OpAsmParser &p,
                            MemoryEffects::Effect *&effect) {
  auto loc = p.getCurrentLocation();
  StringRef name;

  if (p.parseKeyword(name))
    return failure();

  if (!isEffectName(name)) {
    p.emitError(loc, "'").append(name, "' is not a valid memory effect name.");
    return failure();
  }

  effect = getEffect(name);
  return success();
}
} // namespace

void DynMemoryEffectOpInterfaceAttr::print(mlir::OpAsmPrinter &p) const {
  p << "MemoryEffect";
  auto values = getEffects();

  p << "<";
  llvm::interleaveComma(values, p, [&p](MemoryEffects::EffectInstance effect) {
    p << getEffectName(effect.getEffect());
  });
  p << ">";
}

ParseResult
DynMemoryEffectOpInterfaceAttr::parse(mlir::OpAsmParser &p,
                                      DynMemoryEffectOpInterfaceAttr &attr) {
  p.parseLess();

  std::vector<MemoryEffects::Effect *> effects;

  // empty
  if (!p.parseOptionalGreater()) {
    attr = DynMemoryEffectOpInterfaceAttr::get(p.getBuilder().getContext(),
                                               effects);
    return success();
  }

  MemoryEffects::Effect *effect;
  if (parseEffectName(p, effect))
    return failure();
  effects.push_back(effect);

  while (p.parseOptionalGreater()) {
    MemoryEffects::Effect *effect;
    if (p.parseComma() || parseEffectName(p, effect))
      return failure();
    effects.push_back(effect);
  }

  attr =
      DynMemoryEffectOpInterfaceAttr::get(p.getBuilder().getContext(), effects);
  return success();
}

namespace {
void getEffectsConcept(
    const mlir::detail::MemoryEffectOpInterfaceInterfaceTraits::Concept *impl,
    Operation *op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto dynCtx = op->getContext()->getLoadedDialect<dyn::DynamicContext>();
  auto interface = dynCtx->lookupOpInterface(
      mlir::MemoryEffectOpInterface::getInterfaceID());
  assert(succeeded(interface) &&
         "MemoryEffect concept is used but the interface was not declared");

  auto interfaceImplGeneric = (*interface)->getImpl(op);
  assert(succeeded(interfaceImplGeneric) &&
         "MemoryEffect concept is used on an operation that doesn't have an "
         "implementation the interface");

  auto *interfaceImpl =
      reinterpret_cast<DynMemoryEffectOpInterfaceImpl *>(*interfaceImplGeneric);
  for (auto effect : interfaceImpl->effects)
    effects.push_back(effect);
}
} // namespace

void *DynMemoryEffectOpInterface::getConcept() {
  return reinterpret_cast<void *>(
      new (malloc(sizeof(
          mlir::detail::MemoryEffectOpInterfaceInterfaceTraits::Concept)))
          mlir::detail::MemoryEffectOpInterfaceInterfaceTraits::Concept{
              getEffectsConcept});
}

FailureOr<dyn::DynamicOpInterfaceImpl *>
DynMemoryEffectOpInterface::getImpl(Operation *op) {
  auto ctx = op->getContext();
  auto dynCtx = ctx->getLoadedDialect<dyn::DynamicContext>();

  auto dynOpRes = dynCtx->lookupOp(op->getAbstractOperation()->typeID);
  assert(succeeded(dynOpRes) &&
         "Dynamic interfaces can only be used by dynamic operations.");
  auto *dynOp = *dynOpRes;

  return dynOp->getInterfaceImpl(this);
}

ParseResult DynMemoryEffectOpInterface::parseImpl(
    OpAsmParser &p, InterfaceImplAttrInterface &attrInterface) {
  DynMemoryEffectOpInterfaceAttr attr;
  if (DynMemoryEffectOpInterfaceAttr::parse(p, attr))
    return failure();
  attrInterface = attr;
  return success();
}

DynMemoryEffectOpInterfaceImpl::DynMemoryEffectOpInterfaceImpl(
    dyn::DynamicContext *dynCtx, std::vector<MemoryEffects::Effect *> effects)
    : dyn::DynamicOpInterfaceImpl(*dynCtx->lookupOpInterface(
          mlir::MemoryEffectOpInterface::getInterfaceID())),
      effects(std::move(effects)) {}
