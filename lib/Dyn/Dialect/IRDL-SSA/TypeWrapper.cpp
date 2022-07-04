//===- TypeWrapper.cpp - IRDL type wrapper definition -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-SSA/TypeWrapper.h"

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"

namespace mlir {
namespace irdlssa {

DynamicTypeDefinition *findDynamicType(MLIRContext &ctx, StringRef type) {
  auto splitted = type.split('.');
  auto dialectName = splitted.first;
  auto typeName = splitted.second;

  auto dialect = ctx.getOrLoadDialect(dialectName);
  if (!dialect)
    return nullptr;

  auto extensibleDialect = llvm::dyn_cast<ExtensibleDialect>(dialect);
  if (!extensibleDialect)
    return nullptr;

  return extensibleDialect->lookupTypeDefinition(typeName);
}

TypeWrapper *findTypeWrapper(MLIRContext &ctx, StringRef type) {
  Dialect *irdlssaDialect = ctx.getLoadedDialect("irdlssa");
  assert(irdlssaDialect && "irdlssa is not registered");

  IRDLSSADialect *irdlssa = dyn_cast<IRDLSSADialect>(irdlssaDialect);
  assert(irdlssa && "irdlssa dialect is not IRDL-SSA");

  return irdlssa->getTypeWrapper(type);
}

} // namespace irdlssa
} // namespace mlir
