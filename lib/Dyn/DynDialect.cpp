//===- DynDialect.cpp - Dyn dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynDialect.h"
#include "Dyn/DynOps.h"

using namespace mlir;
using namespace mlir::dyn;

//===----------------------------------------------------------------------===//
// Dyn dialect.
//===----------------------------------------------------------------------===//

void DynDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dyn/DynOps.cpp.inc"
      >();
}
