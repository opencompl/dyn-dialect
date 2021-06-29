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

def VarTypeConstraintAttr : IRDL_AttrDef<"VarTypeConstraintAttr", [TypeConstraintAttrInterface]> {
  let summary = "Type constraint variable";
  let description = [{
    Attribute representing a type constraint variable.
    All types matching the variable should be equal.
    The first type matching the variable is the one setting the value.
  }];
  let parameters = (ins "size_t":$index);

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint();
  }];
}

def DynTypeParamsConstraintAttr : IRDL_AttrDef<"DynTypeParamsConstraintAttr", [TypeConstraintAttrInterface]> {
  let summary = "constraint on dynamic type parameters";
  let description = [{
    Attribute representing a constraint on dynamic type parameters.
    The constraint is satisfied by types that are of a specifc base dynamic
    type, and that have parameters satisfying the parameter constraints.
  }];
  let parameters = (ins "DynamicTypeDefinition*":$typeDef,
      ArrayRefParameter<"mlir::Attribute", "list of param constraints">:$paramConstraints);

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint();
  }];
}

class OpDefParameter<string desc = ""> :
    AttrOrTypeParameter<"mlir::irdl::OpDef", desc> {
  let allocator = [{$_dst = mlir::irdl::opDefAllocator($_allocator, $_self);}];
}

def OpDefAttr : IRDL_AttrDef<"OpDefAttr"> {
  let summary = "Declaration of a dynamic operation";
  let description = [{
    Attribute representing an operation definition.
    It contains the arguments/results constraints, as well as trait
    and interface implementation.
  }];
  let parameters = (ins OpDefParameter<>:$opDef);
}

class TypeDefParameter<string desc = ""> :
    AttrOrTypeParameter<"mlir::irdl::TypeDef", desc> {
  let allocator = [{$_dst = mlir::irdl::typeDefAllocator($_allocator, $_self);}];
}

def TypeDefAttr : IRDL_AttrDef<"TypeDefAttr"> {
  let summary = "Declaration of a dynamic operation";
  let description = [{
    Attribute representing an type definition.
    It contains the parameter type constraints.
  }];
  let parameters = (ins TypeDefParameter<>:$typeDef);
}


#endif // DYN_DIALECT_IRDL_IR_IRDLATTRIBUTES