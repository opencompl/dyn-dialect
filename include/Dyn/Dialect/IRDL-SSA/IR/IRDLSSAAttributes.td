//===- IRDLSSAAttributes.td - IRDL-SSA Attributes ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRDL-SSA dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_SSA_IR_IRDLSSAATTRIBUTES
#define DYN_DIALECT_IRDL_SSA_IR_IRDLSSAATTRIBUTES

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "IRDLSSA.td"

class IRDL_SSA_AttrDef<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<IRDL_SSA_Dialect, name, traits, "::mlir::Attribute"> {
  let mnemonic = attrMnemonic;
}

def ParamTypeInstanceAttr : IRDL_SSA_AttrDef<"ParamTypeInstance", "param_type_instance"> {
  let summary = "Attribute representing a specific instance of a possibly parametric type";
  let description = [{
    Attribute representing a specific instance of a possibly parametric type.
  }];
  let parameters = (ins "::mlir::StringAttr":$base, ArrayRefParameter<"::mlir::irdlssa::ParamTypeAttrOrAnyAttr">:$params);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def ParamTypeAttrOrAnyAttr : IRDL_SSA_AttrDef<"ParamTypeAttrOrAny", "param_type_or_any"> {
  let summary = "Attribute representing either a parametric type instance or any attribute";
  let description = [{
    Attribute representing either a parametric type instance or any attribute.
    This is used by `irdlssa.istype` to support more than just built-in types.
  }];
  let parameters = (ins "::mlir::Attribute":$attr);
  let hasCustomAssemblyFormat = 1;
}

#endif // DYN_DIALECT_IRDL_SSA_IR_IRDLSSAATTRIBUTES
