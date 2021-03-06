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
  let description = [{
    Attribute representing the data of an equality type constraint.
    It contains a single type, and the associated type constraint is only
    satisfied when given this type.
  }];
  let parameters = (ins "Type":$type);

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint();
  }];
}

def AnyTypeConstraintAttr : IRDL_AttrDef<"AnyTypeConstraintAttr", [TypeConstraintAttrInterface]> {
  let summary = "Any type constraint attribute";
  let description = [{
    Attribute representing an Any type constraint.
    The associated type constraint is always satisfied.
  }];
  let parameters = (ins);

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint();
  }];
}

def AnyOfTypeConstraintAttr : IRDL_AttrDef<"AnyOfTypeConstraintAttr", [TypeConstraintAttrInterface]> {
  let summary = "AnyOf type constraint attribute";
  let description = [{
    Attribute representing an AnyOf type constraint.
    It contains a set of types.
    The associated type constraint is satisfied when any of the types is given. 
  }];
  let parameters = (ins ArrayRefParameter<"mlir::Type", "list of types">:$types);

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint();
  }];
}

class OpTypeDefParameter<string desc = ""> :
    AttrOrTypeParameter<"mlir::irdl::OpTypeDef", desc> {
  let allocator = [{$_dst = mlir::irdl::opTypeDefAllocator($_allocator, $_self);}];
}

def OpTypeDefAttr : IRDL_AttrDef<"OpTypeDefAttr"> {
  let summary = "Declaration of a dynamic operation";
  let description = [{
    Attribute representing an operation definition.
    It contains the arguments/results constraints, as well as trait
    and interface implementation.
  }];
  let parameters = (ins OpTypeDefParameter<>:$opDef);
}


#endif // DYN_DIALECT_IRDL_IR_IRDLATTRIBUTES