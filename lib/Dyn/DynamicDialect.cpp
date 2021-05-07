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
#include "Dyn/DynamicType.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace dyn;

DynamicDialect::DynamicDialect(llvm::StringRef name, DynamicContext *ctx)
    : DynamicObject{ctx},
      Dialect(name, ctx->getMLIRCtx(), getRuntimeTypeID()), ctx{ctx} {}

DynamicDialect::DynamicDialect(llvm::StringRef name, DynamicContext *ctx,
                               TypeID id)
    : DynamicObject{ctx, id}, Dialect(name, ctx->getMLIRCtx(), id), ctx{ctx} {}

Type DynamicDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef name;

  if (parser.parseKeyword(&name)) {
    parser.emitError(
        typeLoc,
        "dynamic dialect types should be of the format \"dialect.name\"");
    return Type();
  }

  Type typeRes;
  auto parsedDyn =
      DynamicType::parseOptionalDynamicType(this, name, parser, typeRes);

  if (parsedDyn.hasValue()) {
    if (failed(parsedDyn.getValue()))
      return Type();
    return typeRes;
  }

  parser.emitError(typeLoc, "dynamic type '")
      << name << "' was not registered in the dialect " << getNamespace();
  return Type();
}

void DynamicDialect::printType(Type type, DialectAsmPrinter &printer) const {
  auto res = DynamicType::printIfDynamicType(type, printer);
  assert(succeeded(res) &&
         "type registered in dynamic dialect was not dynamic");
}
