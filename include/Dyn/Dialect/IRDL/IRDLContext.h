//===- IRDLContext.h - IRDL context -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the registration context of IRDL dialects.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IRDL_CONTEXT_H_
#define DYN_DIALECT_IRDL_IRDL_CONTEXT_H_

#include "Dyn/Dialect/IRDL/TypeWrapper.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace irdl {

class IRDLContext {
  llvm::StringMap<std::unique_ptr<TypeWrapper>> types;

public:
  IRDLContext() {}

  void addTypeWrapper(std::unique_ptr<TypeWrapper> wrapper);
  TypeWrapper *getTypeWrapper(StringRef typeName);
};

} // namespace irdl
} // namespace mlir

#endif // DYN_DIALECT_IRDL_IRDL_CONTEXT_H_
