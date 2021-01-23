//===- DynamicDialect.cpp - Dynamic dialects --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynamicDialect.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicObject.h"
#include "Dyn/DynamicOperation.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;
using namespace dyn;

DynamicDialect::DynamicDialect(llvm::StringRef name, DynamicContext *ctx)
    : DynamicObject{ctx},
      Dialect(name, ctx->getMLIRCtx(), DynamicObject::getTypeID()), name{name} {
}

DynamicDialect::DynamicDialect(llvm::StringRef name, DynamicContext *ctx,
                               TypeID id)
    : DynamicObject{ctx, id}, Dialect(name, ctx->getMLIRCtx(), id), name{name} {
}

void DynamicDialect::addOperation(DynamicOperation *absOp) {
  AbstractOperation::insert(
      absOp->getName(), *this, {}, absOp->getTypeID(),
      DynamicOperation::parseOperation, DynamicOperation::printOperation,
      DynamicOperation::verifyInvariants, DynamicOperation::foldHook,
      DynamicOperation::getCanonicalizationPatterns,
      detail::InterfaceMap::template get<>(), DynamicOperation::hasTrait);
}

void DynamicDialect::createAndAddOperation(StringRef name) {
  DynamicOperation op(name, this);
  addOperation(&op);
}
