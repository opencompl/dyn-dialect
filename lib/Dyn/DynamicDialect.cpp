//===- DynamicDialect.cpp - dynamic dialects --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynamicDialect.h"
#include "Dyn/DynamicOperation.h"

using namespace mlir;
using namespace dyn;

FailureOr<DynamicOperation *>
DynamicDialect::createAndRegisterOp(llvm::StringRef name) {
  auto op = std::make_unique<DynamicOperation>(DynamicOperation(name, this));
  auto p = ops.try_emplace(name, std::move(op));

  if (!p.second)
    return failure();

  return p.first->second.get();
}
