//===- DynamicContext.h - Dynamic context -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICCONTEXT_H
#define DYN_DYNAMICCONTEXT_H

#include "DynamicID.h"

namespace mlir {
namespace dyn {

/// Manages the creation and lifetime of dynamic MLIR objects such as dialects,
/// operations, types, and traits
class DynamicContext {
public:
  DynamicIDAllocator* getDynamicIDAllocator() {
    return &dynamicIDAllocator;
  }
private:
  DynamicIDAllocator dynamicIDAllocator;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICCONTEXT_H
