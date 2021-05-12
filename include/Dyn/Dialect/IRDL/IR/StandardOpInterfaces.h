//===- StandardOpInterfaces.h - Std interface definitions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the standard interfaces used by IRDL.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_STANDARDOPINTERFACES_H_
#define DYN_DIALECT_IRDL_IR_STANDARDOPINTERFACES_H_

#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
#include "Dyn/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "Dyn/DynamicInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_ATTRDEF_CLASSES
#include "Dyn/Dialect/IRDL/IR/StandardOpInterfaces.h.inc"

namespace mlir {
namespace irdl {

/// Operation interface wrapper for MemoryEffectInterface.
class DynMemoryEffectOpInterface : public dyn::DynamicOpInterface {
public:
  DynMemoryEffectOpInterface(dyn::DynamicContext *ctx)
      : DynamicOpInterface(ctx, mlir::MemoryEffectOpInterface::getInterfaceID(),
                           "MemoryEffect") {}

  virtual void *getConcept() override;

  virtual ParseResult parseImpl(OpAsmParser &p,
                                InterfaceImplAttrInterface &interface) override;
};

/// Implementation of a MemoryEffectInterface.
class DynMemoryEffectOpInterfaceImpl : public dyn::DynamicOpInterfaceImpl {
public:
  DynMemoryEffectOpInterfaceImpl(dyn::DynamicContext *dynCtx,
                                 std::vector<MemoryEffects::Effect *> effects);

  const std::vector<MemoryEffects::Effect *> effects;
};

} // namespace irdl
} // namespace mlir

#endif //  DYN_DIALECT_IRDL_IR_STANDARDOPINTERFACES_H_
