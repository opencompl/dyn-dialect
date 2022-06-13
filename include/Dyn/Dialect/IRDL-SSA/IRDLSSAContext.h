//===- IRDLSSAContext.h - IRDL-SSA context ----------------------*- C++ -*-===//
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

#ifndef DYN_DIALECT_IRDL_SSA_IRDLSSA_CONTEXT_H_
#define DYN_DIALECT_IRDL_SSA_IRDLSSA_CONTEXT_H_

#include "Dyn/Dialect/IRDL-SSA/TypeWrapper.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace irdlssa {

/// Context for the runtime registration of IRDL dialect definitions.
class IRDLSSAContext {
  llvm::StringMap<std::unique_ptr<TypeWrapper>> types;

public:
  IRDLSSAContext() {}

  void addTypeWrapper(std::unique_ptr<TypeWrapper> wrapper);
  TypeWrapper *getTypeWrapper(StringRef typeName);

  llvm::StringMap<std::unique_ptr<TypeWrapper>> const &getAllTypes() {
    return this->types;
  }
};

/// Context for the analysis of IRDL dialect definitions.
struct TypeContext {
  /// Utility constructor to load all type information from an IRDLSSAContext.
  TypeContext(IRDLSSAContext &ctx);

  struct TypeInfo {
    size_t paramAmount;
    TypeInfo(size_t paramAmount) : paramAmount(paramAmount) {}
  };

  llvm::StringMap<TypeInfo> types;
};

} // namespace irdlssa
} // namespace mlir

#endif // DYN_DIALECT_IRDL_SSA_IRDLSSA_CONTEXT_H_
