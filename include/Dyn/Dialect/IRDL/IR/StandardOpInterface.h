//===- StandardOpInterfaceAttrs.h - Std Interface Attributes ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes used in the IRDL dialect to represent
// standard interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_STANDARDOPINTERFACEATTRS_H_
#define DYN_DIALECT_IRDL_IR_STANDARDOPINTERFACEATTRS_H_

#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
#include "Dyn/Dialect/IRDL/IR/IRDLInterface.h"
#include "Dyn/DynamicInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace irdl {

/// Attribute used in IRDL to represent the implementation of a
/// MemoryEffectInterface.
class DynMemoryEffectOpInterfaceAttr
    : public mlir::Attribute::AttrBase<
          DynMemoryEffectOpInterfaceAttr, mlir::Attribute,
          detail::StringArrayAttrStorage, InterfaceImplAttrInterface::Trait> {
public:
  // Using Attribute constructors.
  using Base::Base;

  static DynMemoryEffectOpInterfaceAttr get(MLIRContext &ctx,
                                            ArrayRef<StringRef> effects) {
    return Base::get(&ctx, effects);
  }

  std::unique_ptr<mlir::dyn::DynamicOpInterfaceImpl> getInterfaceImpl();

  void print(mlir::OpAsmPrinter &p);
  static ParseResult parse(mlir::OpAsmParser &p,
                           DynMemoryEffectOpInterfaceAttr &attr);
};

/// Operation interface wrapper for MemoryEffectInterface.
class DynMemoryEffectOpInterface : public dyn::DynamicOpInterface {
public:
  DynMemoryEffectOpInterface(dyn::DynamicContext *ctx)
      : DynamicOpInterface(ctx, mlir::MemoryEffectOpInterface::getInterfaceID(),
                           "MemoryEffect") {}

  virtual void *getConcept() override;

  virtual FailureOr<dyn::DynamicOpInterfaceImpl *>
  getImpl(Operation *op) override;

  virtual ParseResult parseImpl(OpAsmParser &p,
                                InterfaceImplAttrInterface &interface) override;
};

/// Implementation of a MemoryEffectInterface.
class DynMemoryEffectOpInterfaceImpl : public dyn::DynamicOpInterfaceImpl {
public:
  DynMemoryEffectOpInterfaceImpl(
      dyn::DynamicContext *dynCtx,
      std::vector<MemoryEffects::EffectInstance> effects);

  const std::vector<MemoryEffects::EffectInstance> effects;
};

} // namespace irdl
} // namespace mlir

#endif //  DYN_DIALECT_IRDL_IR_STANDARDOPINTERFACEATTRS_H_
