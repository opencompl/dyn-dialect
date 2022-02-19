//===- IRDLOps.td - IR Definition Language Dialect ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRDL dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_IRDLOPS
#define DYN_DIALECT_IRDL_IR_IRDLOPS

include "IRDL.td"
include "IRDLAttributes.td"
include "mlir/IR/SymbolInterfaces.td"

class IRDL_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<IRDL_Dialect, mnemonic, traits> {
}

class AtMostOneChildOf<string op> : ParamNativeOpTrait<"AtMostOneChildOf", op>;

//===----------------------------------------------------------------------===//
// IRDL Dialect definition
//===----------------------------------------------------------------------===//

def IRDL_DialectOp : IRDL_Op<"dialect", [
    IsolatedFromAbove, SingleBlockImplicitTerminator<"EndDialectOp">
  ]> {
  let summary = "Define a new dialect";
  let description = [{
    `irdl.dialect` operations defines a dialect.

    Example:

    ```mlir
    irdl.dialect "cmath" {
      ...
    }
    ```

    The above program defines a `cmath` dialect.
  }];

  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let printer = [{ ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let verifier = [{ return ::verify(*this); }];
}

def IRDL_EndDialectOp : IRDL_Op<"end_dialect", [Terminator]> {
  let summary = "Implicit terminator of `irdl.dialect` operation";
}

//===----------------------------------------------------------------------===//
// IRDL Type and Attribute parameters definition
//===----------------------------------------------------------------------===//

def IRDL_ParametersOp : IRDL_Op<"parameters", [HasParent<"TypeOp">]> {
  let summary = "Define the parameters of a type/attribute definition";
  let description = [{
    `irdl.parameters` defines new parameters belonging to the type or attribute
    defined by the parent operation.

    Example:

    ```mlir
    irdl.dialect "cmath" {
      irdl.type "complex" {
        irdl.parameters(elementType: AnyOf<i32, i64>)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath`. The
    type has a single parameter named `elementType` that should be either `i32`
    or `i64`.
  }];

  let arguments = (ins TypedArrayAttrBase<NamedTypeConstraintAttr, "Named constraint array">:$params);
  let assemblyFormat = [{ `` custom<NamedTypeConstraintArray>($params) attr-dict }];
}

//===----------------------------------------------------------------------===//
// IRDL Type definition
//===----------------------------------------------------------------------===//

def IRDL_TypeOp : IRDL_Op<"type", [HasParent<"DialectOp">, NoTerminator, NoRegionArguments, AtMostOneChildOf<"ParametersOp">]> {
  let summary = "Define a new type";
  let description = [{
    `irdl.type` defines new types belonging to the previously defined dialect
    using `irdl.dialect`.

    The type parameters can be defined with an `irdl.parameters` operation in
    the optional region.

    Example:

    ```mlir
    irdl.dialect "cmath" {
      irdl.type "complex" {
        irdl.parameters(elementType: AnyOf<i32, i64>)
      }
    }
    ```
    
    The above program defines a type `complex` inside the dialect `cmath`. The
    type has a single parameter that should be either `i32` or `i64`.
  }];

  let arguments = (ins TypeDefAttr:$def);
  let regions = (region SizedRegion<1>:$body);
  let printer = [{ ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let extraClassDeclaration = [{
    /// Get the parent dialect operation.
    DialectOp getDialectOp() { return cast<DialectOp>(getOperation()->getParentOp()); };
  }];
}

//===----------------------------------------------------------------------===//
// IRDL Operation definition
//===----------------------------------------------------------------------===//


def IRDL_OperationOp : IRDL_Op<"operation", [HasParent<"DialectOp">]> {
  let summary = "Define a new operation";
  let description = [{
    `irdl.operation` defines new operations belonging to the previously
    defined dialect using `irdl.dialect`.

    Those operations have no verifiers.

    Example:

    ```mlir
    irdl.dialect "cmath" {
      irdl.operation "mul"
    }
    ```
    
    The above program defines an operation `mul` inside the dialect `cmath`.
  }];

  let arguments = (ins StrAttr:$name, OpDefAttr:$op_def);
  let printer = [{ ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let extraClassDeclaration = [{
    /// Get the parent dialect operation.
    DialectOp getDialectOp() { return cast<DialectOp>(getOperation()->getParentOp()); };
  }];
}

#endif // DYN_DIALECT_IRDL_IR_IRDLOPS