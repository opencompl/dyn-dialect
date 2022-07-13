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

/// Context for the runtime registration of IRDL dialect definitions.
class IRDLContext {
  llvm::StringMap<std::unique_ptr<TypeWrapper>> types;

public:
  IRDLContext() {}

  void addTypeWrapper(std::unique_ptr<TypeWrapper> wrapper);
  TypeWrapper *getTypeWrapper(StringRef typeName);

  llvm::StringMap<std::unique_ptr<TypeWrapper>> const &getAllTypes() {
    return this->types;
  }
};

/// Context for the analysis of IRDL dialect definitions.
struct TypeContext {
  /// Utility constructor to load all type information from an IRDLContext.
  TypeContext(IRDLContext &ctx);

  struct TypeInfo {
    size_t paramAmount;
    TypeInfo(size_t paramAmount) : paramAmount(paramAmount) {}
  };

  llvm::StringMap<TypeInfo> types;
};

} // namespace irdlssa
} // namespace mlir

#endif // DYN_DIALECT_IRDL_IRDL_CONTEXT_H_
