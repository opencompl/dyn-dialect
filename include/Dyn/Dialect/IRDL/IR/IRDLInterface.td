//===- IRDLInterface.td - IRDL Attributes ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the interfaces IRDL uses.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_IRDLATTRIBUTES
#define DYN_DIALECT_IRDL_IR_IRDLATTRIBUTES

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def TypeConstraintAttrInterface : AttrInterface<"TypeConstraintAttrInterface"> {
  let description = [{
    Attribute representing a constraint on a dynamic type.
  }];

  let methods = [
    InterfaceMethod<
      "Get the type constraint",
      "std::unique_ptr<mlir::irdl::TypeConstraint>", "getTypeConstraint",
      (ins "mlir::dyn::DynamicContext &":$ctx)
    >
  ];
}

def InterfaceImplAttrInterface : AttrInterface<"InterfaceImplAttrInterface"> {
  let description = [{
    Attribute representing the implementation of an interface.
  }];

  let methods = [
    InterfaceMethod<
      "Get the interface implementation",
      "std::unique_ptr<mlir::dyn::DynamicOpInterfaceImpl>", "getInterfaceImpl"
    >,
    InterfaceMethod<
      "Print the attribute",
      "void", "print", (ins "mlir::OpAsmPrinter&":$p)
    >
  ];
}

#endif // DYN_DIALECT_IRDL_IR_IRDLATTRIBUTES