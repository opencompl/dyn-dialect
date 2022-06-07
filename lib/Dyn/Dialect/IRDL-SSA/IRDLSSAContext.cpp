//===- IRDLSSAContext.cpp - IRDL-SSA context --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-SSA/IRDLSSAContext.h"

using namespace mlir;
using namespace irdlssa;

TypeContext::TypeContext(IRDLSSAContext &ctx) {
  for (auto &wrapper : ctx.getAllTypes()) {
    this->types.insert(
        {wrapper.getKey(), TypeInfo(wrapper.getValue()->getParameterAmount())});
  }
}

void IRDLSSAContext::addTypeWrapper(std::unique_ptr<TypeWrapper> wrapper) {
  auto emplaced =
      this->types.try_emplace(wrapper->getName(), std::move(wrapper)).second;
  assert(emplaced && "a type wrapper with the same name already exists");
}

TypeWrapper *IRDLSSAContext::getTypeWrapper(StringRef typeName) {
  auto it = this->types.find(typeName);
  if (it == this->types.end())
    return nullptr;
  return it->second.get();
}
