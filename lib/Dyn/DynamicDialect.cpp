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

Type DynamicDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef name;

  if (parser.parseKeyword(&name)) {
    parser.emitError(
        typeLoc,
        "dynamic dialect types should be of the format \"dialect.name\"");
    return Type();
  }

  auto fullName = (getName() + "." + name).str();

  auto type = getDynamicContext()->lookupType(fullName);
  if (succeeded(type))
    return DynamicType::get(ctx->getMLIRCtx(), *type);

  auto alias = getDynamicContext()->lookupTypeAlias(fullName);
  if (succeeded(alias))
    return *alias;

  parser.emitError(typeLoc, "dynamic type '")
      << name << "' was not registered in the dialect " << getName();
  return Type();
}

void DynamicDialect::printType(Type type, DialectAsmPrinter &printer) const {
  auto dynType = getDynamicContext()->lookupType(type.getTypeID());
  assert(!failed(dynType));
  printer << (*dynType)->name;
}
