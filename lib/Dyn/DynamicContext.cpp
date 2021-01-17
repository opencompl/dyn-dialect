//===- DynamicContext.cpp - Dynamic context ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"

using namespace mlir;
using namespace dyn;

mlir::FailureOr<DynamicDialect *>
DynamicContext::createAndRegisterDialect(llvm::StringRef name) {
  auto dialect = std::make_unique<DynamicDialect>(name);
  auto p = dialects.try_emplace(name, std::move(dialect));

  if (!p.second)
    return failure();

  return p.first->second.get();
}
