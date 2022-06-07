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

class IRDL_Op<string mnemonic, list<Trait> traits = []>
    : Op<IRDL_Dialect, mnemonic, traits> {
}

class AtMostOneChildOf<string op> : ParamNativeOpTrait<"AtMostOneChildOf", op>;

//===----------------------------------------------------------------------===//
// IRDL Dialect definition
//===----------------------------------------------------------------------===//

def IRDL_DialectOp : IRDL_Op<"dialect", [IsolatedFromAbove, NoTerminator]> {
  let summary = "Define a new dialect";
  let description = [{
    `irdl.dialect` operations defines a dialect.

    Example:

    ```mlir
    irdl.dialect cmath {
      ...
    }
    ```

    The above program defines a `cmath` dialect.
  }];

  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "custom<KeywordOrString>($name) attr-dict-with-keyword custom<SingleBlockRegion>($body)";
  let hasVerifier = 1;
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
    irdl.dialect cmath {
      irdl.type complex {
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

def IRDL_TypeOp : IRDL_Op<"type", [HasParent<"DialectOp">, NoTerminator, NoRegionArguments,
                                   AtMostOneChildOf<"ParametersOp, ConstraintVarsOp">]> {
  let summary = "Define a new type";
  let description = [{
    `irdl.type` defines new types belonging to the previously defined dialect
    using `irdl.dialect`.

    The type parameters can be defined with an `irdl.parameters` operation in
    the optional region.

    Example:

    ```mlir
    irdl.dialect cmath {
      irdl.type complex {
        irdl.parameters(elementType: AnyOf<i32, i64>)
      }
    }
    ```
    
    The above program defines a type `complex` inside the dialect `cmath`. The
    type has a single parameter that should be either `i32` or `i64`.
  }];

  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "custom<KeywordOrString>($name) attr-dict-with-keyword custom<SingleBlockRegion>($body)";  

  let extraClassDeclaration = [{
    /// Get the parent dialect operation.
    DialectOp getDialectOp() { return cast<DialectOp>(getOperation()->getParentOp()); };
  }];
}

//===----------------------------------------------------------------------===//
// IRDL Operation definition
//===----------------------------------------------------------------------===//

def IRDL_ConstraintVarsOp : IRDL_Op<"constraint_vars", [HasParent<"OperationOp">]> {
  let summary = "Define constraint variables";
  let description = [{
    `irdl.constraint_vars` define constraint variables that can be used in the
    current region.

    Example:

    ```mlir
    irdl.dialect testd {
      irdl.operation test_op {
        irdl.constraint_vars(T: irdl.AnyOf<i32, i64>)
        irdl.operands(operand1: ?T, operand2: ?T)
      }
    }
    ```

    The above program defines an operation `test_op` with two expected operands.
    Both operands are expected to be either of type `i32` or of type `i64`.
    Additionally, they are expected to be of equal type.
  }];

  let arguments = (ins TypedArrayAttrBase<NamedTypeConstraintAttr, "Named constraint array">:$params);
  let assemblyFormat = [{ `` custom<NamedTypeConstraintArray>($params) attr-dict }];
}


def IRDL_OperandsOp : IRDL_Op<"operands", [HasParent<"OperationOp">]> {
  let summary = "Define the operands of an operation";
  let description = [{
    `irdl.operands` define the operands of the `irdl.operation` parent operation
    definition.

    Example:

    ```mlir
    irdl.dialect testd {
      irdl.operation test_op {
        irdl.operands(operand1: !i32, operand2: AnyOf<i32, i64>)
      }
    }
    ```

    The above program defines an operation `test_op` with two expected operands.
    The first operand is expected to be of type `i32`, and the second operand is
    expected to be either of type `i32` or `i64`.
  }];

  let arguments = (ins TypedArrayAttrBase<NamedTypeConstraintAttr, "Named constraint array">:$params);
  let assemblyFormat = [{ `` custom<NamedTypeConstraintArray>($params) attr-dict }];
}

def IRDL_ResultsOp : IRDL_Op<"results", [HasParent<"OperationOp">]> {
  let summary = "Define the results of an operation";
  let description = [{
    `irdl.results` define the results of the `irdl.operation` parent operation
    definition.

    Example:

    ```mlir
    irdl.dialect testd {
      irdl.operation test_op {
        irdl.results(result1: !i32, result2: AnyOf<i32, i64>)
      }
    }
    ```

    The above program defines an operation `test_op` with two expected results.
    The first result is expected to be an `i32`, and the second result is
    expected to be either an `i32` or an `i64`.
  }];

  let arguments = (ins TypedArrayAttrBase<NamedTypeConstraintAttr, "Named constraint array">:$params);
  let assemblyFormat = [{ `` custom<NamedTypeConstraintArray>($params) attr-dict }];
}

def IRDL_OperationOp : IRDL_Op<"operation",
    [HasParent<"DialectOp">, NoTerminator, NoRegionArguments,
    AtMostOneChildOf<"OperandsOp, ResultsOp, ConstraintVarsOp">]> {
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

  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "custom<KeywordOrString>($name) attr-dict-with-keyword custom<SingleBlockRegion>($body)";

  let extraClassDeclaration = [{
    /// Get the parent dialect operation.
    DialectOp getDialectOp() { return cast<DialectOp>(getOperation()->getParentOp()); };
  }];
}

#endif // DYN_DIALECT_IRDL_IR_IRDLOPS