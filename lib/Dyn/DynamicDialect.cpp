//===- DynamicDialect.cpp - Dynamic dialects --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynamicDialect.h"
#include "Dyn/DynamicOperation.h"
#include "Dyn/DynamicContext.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;
using namespace dyn;

DynamicDialect::DynamicDialect(llvm::StringRef name, DynamicContext *ctx)
    : DynamicObject{ctx},
      Dialect(name, ctx->getMLIRCtx(), DynamicObject::getTypeID()), name{name} {
}

DynamicDialect::DynamicDialect(llvm::StringRef name, DynamicContext *ctx, TypeID id)
  : DynamicObject{ctx, id},
    Dialect(name, ctx->getMLIRCtx(), id), name{name} {
}

FailureOr<DynamicOperation *>
DynamicDialect::createAndRegisterOp(llvm::StringRef name) {
  auto op = std::make_unique<DynamicOperation>(DynamicOperation(name, this));
  auto p = ops.try_emplace(name, std::move(op));

  if (!p.second)
    return failure();

  return p.first->second.get();
}
