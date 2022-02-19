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

class IRDL_AttrDef<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<IRDL_Dialect, name, traits, "::mlir::Attribute"> {
  let mnemonic = attrMnemonic;
}

def EqTypeConstraintAttr : IRDL_AttrDef<"EqTypeConstraint", "eq_type_constraint",
                                        [TypeConstraintAttrInterface]> {
  let summary = "IRDL equality type constraint attribute";
  let description = [{
    Attribute representing the data of an equality type constraint.
    It contains a single type, and the associated type constraint is only
    satisfied when given this type.
  }];
  let parameters = (ins "Type":$type);
  let assemblyFormat = "`<` $type `>`";

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint();
  }];
}

def AnyTypeConstraintAttr : IRDL_AttrDef<"AnyTypeConstraint", "any_type_constraint",
                                         [TypeConstraintAttrInterface]> {
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

def AnyOfTypeConstraintAttr : IRDL_AttrDef<"AnyOfTypeConstraint", "any_of_type_constraint",
                                           [TypeConstraintAttrInterface]> {
  let summary = "AnyOf type constraint attribute";
  let description = [{
    Attribute representing an AnyOf type constraint.
    It contains a set of types.
    The associated type constraint is satisfied when any of the types is given. 
  }];
  let parameters = (ins ArrayRefParameter<"mlir::Type", "list of types">:$types);
  let assemblyFormat = "`<` $types `>`";

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint();
  }];
}

def VarTypeConstraintAttr : IRDL_AttrDef<"VarTypeConstraint", "var_type_constraint",
                                         [TypeConstraintAttrInterface]> {
  let summary = "Type constraint variable";
  let description = [{
    Attribute representing a type constraint variable.
    All types matching the variable should be equal.
    The first type matching the variable is the one setting the value.
  }];
  let parameters = (ins "size_t":$index);
  let assemblyFormat = "`<` $index `>`";

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint();
  }];
}

def DynTypeParamsConstraintAttr : IRDL_AttrDef<"DynTypeParamsConstraint", "dyn_type_params_constraint",
                                               [TypeConstraintAttrInterface]> {
  let summary = "constraint on dynamic type parameters";
  let description = [{
    Attribute representing a constraint on dynamic type parameters.
    The constraint is satisfied by types that are of a specifc base dynamic
    type, and that have parameters satisfying the parameter constraints.
  }];
  let parameters = (ins StringRefParameter<>:$typeName,
      ArrayRefParameter<"mlir::Attribute", "list of param constraints">:$paramConstraints);
  let assemblyFormat = "`<` $typeName `<` $paramConstraints `>` `>`";

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint();
  }];
}

class TypeWrapperParameter<string desc = ""> :
    AttrOrTypeParameter<"mlir::irdl::TypeWrapper*", desc> {
  let printer = "assert(false)";
  let parser = "failure()";
}

def TypeParamsConstraintAttr : IRDL_AttrDef<"TypeParamsConstraint", "type_params_constraint",
                                            [TypeConstraintAttrInterface]> {
  let summary = "constraint on C++ type parameters";
  let description = [{
    Attribute representing a constraint on C++ type parameters.
    The constraint is satisfied by types that are of a specifc base
    type, and that have parameters satisfying the parameter constraints.
  }];
  let parameters = (ins TypeWrapperParameter<>:$typeDef,
      ArrayRefParameter<"mlir::Attribute", "list of param constraints">:$paramConstraints);
  let assemblyFormat = "`<` $typeDef `<` $paramConstraints `>` `>`";

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint();
  }];
}

def TypeConstraintParameter :
  AttrOrTypeParameter<"::mlir::Attribute", "type constraint attribute"> {}

def NamedTypeConstraintAttr : IRDL_AttrDef<"NamedTypeConstraint",
                                           "Named type constraint"> {
  let summary = "Type constraint with an associated name";
  let parameters = (ins StringRefParameter<>:$name, TypeConstraintParameter:$constraint);
  let assemblyFormat = "`<` $name `:` $constraint `>`";
}

class OpDefParameter<string desc = ""> :
    AttrOrTypeParameter<"mlir::irdl::OpDef", desc> {
  let allocator = [{$_dst = mlir::irdl::opDefAllocator($_allocator, $_self);}];
  let printer = "assert(false)";
  let parser = "failure()";
}

def OpDefAttr : IRDL_AttrDef<"OpDef", "op_def"> {
  let summary = "Declaration of a dynamic operation";
  let description = [{
    Attribute representing an operation definition.
    It contains the arguments/results constraints, as well as trait
    and interface implementation.
  }];
  let parameters = (ins OpDefParameter<>:$opDef);
  let assemblyFormat = "`<` $opDef `>`";
}

#endif // DYN_DIALECT_IRDL_IR_IRDLATTRIBUTES