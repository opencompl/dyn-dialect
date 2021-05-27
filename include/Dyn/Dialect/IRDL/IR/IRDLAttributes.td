//===- IRDLAttributes.td - IRDL Attributes -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes IRDL uses.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_IRDLATTRIBUTES
#define DYN_DIALECT_IRDL_IR_IRDLATTRIBUTES

include "mlir/IR/OpBase.td"
include "IRDLInterfaces.td"
include "IRDL.td"

def EqTypeConstraintAttr : IRDL_AttrDef<"EqTypeConstraintAttr", [TypeConstraintAttrInterface]> {
  let summary = "IRDL equality type constraint attribute";
  let description = [{ TODO }];
  let parameters = (ins "Type":$type);

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint(mlir::dyn::DynamicContext &ctx);
  }];
}

def AnyTypeConstraintAttr : IRDL_AttrDef<"AnyTypeConstraintAttr", [TypeConstraintAttrInterface]> {
  let summary = "Any type constraint attribute";
  let description = [{ TODO }];
  let parameters = (ins);

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint(mlir::dyn::DynamicContext &ctx);
  }];
}

def AnyOfTypeConstraintAttr : IRDL_AttrDef<"AnyOfTypeConstraintAttr", [TypeConstraintAttrInterface]> {
  let summary = "AnyOf type constraint attribute";
  let description = [{ TODO }];
  let parameters = (ins ArrayRefParameter<"mlir::Type", "list of types">:$types);

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint(mlir::dyn::DynamicContext &ctx);
  }];
}

class OpTypeDefParameter<string desc = ""> :
    AttrOrTypeParameter<"mlir::irdl::OpTypeDef", desc> {
  let allocator = [{$_dst = mlir::irdl::opTypeDefAllocator($_allocator, $_self);}];
}

def OpTypeDefAttr : IRDL_AttrDef<"OpTypeDefAttr"> {
  let summary = "Declaration of a dynamic operation";
  let description = [{ TODO }];
  let parameters = (ins OpTypeDefParameter<>:$opDef);
}


#endif // DYN_DIALECT_IRDL_IR_IRDLATTRIBUTES