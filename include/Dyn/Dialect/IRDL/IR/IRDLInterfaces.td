//===- IRDLInterfaces.td - IRDL Attributes -----------------*- tablegen -*-===//
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

#ifndef DYN_DIALECT_IRDL_IR_IRDLINTERFACES
#define DYN_DIALECT_IRDL_IR_IRDLINTERFACES

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def TypeConstraintAttrInterface : AttrInterface<"TypeConstraintAttrInterface"> {
  let cppNamespace = "::mlir::irdl";

  let description = [{
    Attribute representing a constraint on a dynamic type.
  }];

  let methods = [
    InterfaceMethod<
      "Convert the constraint to SSA",
      "mlir::Value", "registerAsSSA",
      (ins "mlir::irdlssa::TypeContext &":$ctx,
           "mlir::ConversionPatternRewriter &":$rewriter,
           "mlir::SmallVector<std::pair<llvm::StringRef, mlir::Value>> &":$vars,
           "mlir::Location":$location)
    >
  ];
}

#endif // DYN_DIALECT_IRDL_IR_IRDLINTERFACES