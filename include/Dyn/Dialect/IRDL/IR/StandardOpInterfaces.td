//===- StandardOpInterfaces.td - Std interface attributes --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes used to define the standard interface
// implementations in IRDL.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_STANDARDOPINTERFACES
#define DYN_DIALECT_IRDL_IR_STANDARDOPINTERFACES

include "mlir/IR/OpBase.td"
include "IRDL.td"
include "IRDLInterfaces.td"

def DynMemoryEffectOpInterfaceAttr : IRDL_AttrDef<"DynMemoryEffectOpInterfaceAttr", [InterfaceImplAttrInterface]> {
  let summary = "Represent the implementation of a MemoryEffectInterface";
  let description = [{ TODO }];
  let parameters = (ins ArrayRefParameter<"MemoryEffects::Effect*">:$effects);

  let parser = [{ return ::parse$cppClass(parser, result); }];

  let extraClassDeclaration = [{
    /// Get the interface implementation.
    std::unique_ptr<mlir::dyn::DynamicOpInterfaceImpl> getInterfaceImpl() const;
    /// Print the attribute.
    void print(mlir::OpAsmPrinter& p) const;
    /// Parse the attribute.
    static ParseResult parse(mlir::OpAsmParser &p, mlir::irdl::DynMemoryEffectOpInterfaceAttr &attrInterface);
  }];

}

#endif // DYN_DIALECT_IRDL_IR_STANDARDOPINTERFACES
