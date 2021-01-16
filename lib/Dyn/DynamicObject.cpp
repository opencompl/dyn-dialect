//===- DynamicObject.cpp - Dynamic dialect ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynamicObject.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicID.h"

namespace mlir {
namespace dyn {

  DynamicObject::DynamicObject(DynamicContext *ctx) :
    ctx{ctx},
    dynamicID{ctx->getDynamicIDAllocator()->allocateID()}
  {}

}
}
