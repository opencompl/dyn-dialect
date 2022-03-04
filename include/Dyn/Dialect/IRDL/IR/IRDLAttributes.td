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
    std::unique_ptr<TypeConstraint> getTypeConstraint(
        SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const& constr_vars);
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
    std::unique_ptr<TypeConstraint> getTypeConstraint(
        SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const& constr_vars);
  }];
}

def AnyOfTypeConstraintAttr : IRDL_AttrDef<"AnyOfTypeConstraint", "any_of_type_constraint",
                                           [TypeConstraintAttrInterface]> {
  let summary = "AnyOf type constraint attribute";
  let description = [{
    Attribute representing an AnyOf type constraint.
    It contains a set of constraints.
    The associated type constraint is satisfied when any of the constraints is satisfied. 
  }];
  let parameters = (ins ArrayRefParameter<"mlir::Attribute", "list of constraints">:$constrs);
  let assemblyFormat = "`<` $constrs `>`";

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint(
        SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const& constr_vars);
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
  let parameters = (ins StringRefParameter<>:$name);
  let assemblyFormat = "`<` $name `>`";

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint(
        SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const& constr_vars);
  }];
}

def DynTypeBaseConstraintAttr : IRDL_AttrDef<"DynTypeBaseConstraint", "dyn_type_base_constraint",
                                               [TypeConstraintAttrInterface]> {
  let summary = "constraint on the base of a dynamic type";
  let description = [{
    Attribute representing a constraint on dynamic type base type.
    The constraint is satisfied by types that have the correct base type.
  }];
  let parameters = (ins StringRefParameter<>:$typeName);
  let assemblyFormat = "`<` $typeName `>`";

  let extraClassDeclaration = [{
    std::unique_ptr<TypeConstraint> getTypeConstraint(
        SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const& constr_vars);
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
    std::unique_ptr<TypeConstraint> getTypeConstraint(
        SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const& constr_vars);
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
    std::unique_ptr<TypeConstraint> getTypeConstraint(
        SmallVector<std::pair<StringRef, std::unique_ptr<TypeConstraint>>> const& constr_vars);
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

#endif // DYN_DIALECT_IRDL_IR_IRDLATTRIBUTES