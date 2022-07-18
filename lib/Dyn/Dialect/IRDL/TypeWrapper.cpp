//===- TypeWrapper.cpp - IRDL type wrapper definition -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/TypeWrapper.h"

#include "Dyn/Dialect/IRDL/IR/IRDL.h"

namespace mlir {
namespace irdl {

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
  IRDLDialect *irdl = ctx.getLoadedDialect<IRDLDialect>();
  assert(irdl && "irdl is not registered");

  return irdl->getTypeWrapper(type);
}

} // namespace irdl
} // namespace mlir
