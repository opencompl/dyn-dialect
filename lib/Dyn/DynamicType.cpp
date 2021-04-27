//===- DynamicType.cpp - Dynamic types --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represent types that can be defined at runtime.
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynamicType.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace dyn;

DynamicTypeDefinition::DynamicTypeDefinition(Dialect *dialect,
                                             llvm::StringRef name,
                                             VerifierFn verifier)
    : DynamicObject(dialect->getContext()->getLoadedDialect<DynamicContext>()),
      dialect(dialect), name(name.str()), verifier(std::move(verifier)) {}

DynamicTypeDefinition *DynamicType::getTypeDef() { return getImpl()->typeDef; }

DynamicType DynamicType::get(DynamicTypeDefinition *typeDef,
                             ArrayRef<Attribute> params) {
  auto ctx = typeDef->getDynamicContext()->getMLIRCtx();
  return detail::TypeUniquer::getWithTypeID<DynamicType>(
      ctx, typeDef->getRuntimeTypeID(), typeDef, params);
}

DynamicType
DynamicType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        DynamicTypeDefinition *typeDef,
                        ArrayRef<Attribute> params) {
  if (failed(typeDef->verify(emitError, params)))
    return {};
  return get(typeDef, params);
}

ParseResult DynamicType::parse(DialectAsmParser &parser,
                               DynamicTypeDefinition *typeDef,
                               DynamicType &parsedType) {
  auto loc = parser.getCurrentLocation();
  auto emitError = [&]() { return parser.emitError(loc); };

  if (parser.parseOptionalLess() || !parser.parseOptionalGreater()) {
    parsedType = DynamicType::getChecked(emitError, typeDef, {});
    return success();
  }

  std::vector<Attribute> attrs;
  attrs.push_back({});
  if (parser.parseAttribute(attrs.back()))
    return failure();

  while (parser.parseOptionalGreater()) {
    attrs.push_back({});
    if (parser.parseComma() || parser.parseAttribute(attrs.back()))
      return failure();
  }

  parsedType = DynamicType::getChecked(emitError, typeDef, attrs);
  return success();
}

void DynamicType::print(DialectAsmPrinter &printer) {
  printer << getTypeDef()->name;
  auto params = getImpl()->params;

  if (params.empty())
    return;

  printer << "<";
  llvm::interleaveComma(params, printer.getStream());
  printer << ">";
}

OptionalParseResult DynamicType::parseOptionalDynamicType(
    const Dialect *dialect, StringRef typeName, DialectAsmParser &parser,
    Type &resultType) {
  auto fullName = (dialect->getNamespace() + "." + typeName).str();

  auto dynCtx = dialect->getContext()->getLoadedDialect<DynamicContext>();

  auto alias = dynCtx->lookupTypeAlias(fullName);
  if (succeeded(alias)) {
    resultType = *alias;
    return {success()};
  }

  auto typeDef = dynCtx->lookupTypeDefinition(fullName);
  if (succeeded(typeDef)) {
    DynamicType dynType;
    if (DynamicType::parse(parser, *typeDef, dynType))
      return failure();
    resultType = dynType;
    return {success()};
  }

  return {};
}

LogicalResult DynamicType::printIfDynamicType(Type type,
                                              DialectAsmPrinter &printer) {
  auto dynTypeDef = type.getContext()
                        ->getLoadedDialect<DynamicContext>()
                        ->lookupTypeDefinition(type.getTypeID());
  if (failed(dynTypeDef))
    return failure();

  auto dynType = type.cast<DynamicType>();
  dynType.print(printer);
  return success();
}

bool DynamicType::classof(Type type) {
  return succeeded(type.getContext()
                       ->getLoadedDialect<DynamicContext>()
                       ->lookupTypeDefinition(type.getTypeID()));
}
