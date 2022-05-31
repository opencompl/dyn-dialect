//===- IRDLSSAOps.td - IR Definition Language Dialect ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRDL-SSA dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_SSA_IR_IRDLSSAOPS
#define DYN_DIALECT_IRDL_SSA_IR_IRDLSSAOPS

include "IRDLSSA.td"
include "IRDLSSATypes.td"
include "IRDLSSAInterfaces.td"

class IRDL_SSA_Op<string mnemonic, list<Trait> traits = []>
    : Op<IRDL_SSA_Dialect, mnemonic, traits> {
}

class AtMostOneChildOf<string op> : ParamNativeOpTrait<"AtMostOneChildOf", op>;

//===----------------------------------------------------------------------===//
// IRDL-SSA Dialect definition
//===----------------------------------------------------------------------===//

def IRDL_SSA_DialectOp : IRDL_SSA_Op<"dialect", [IsolatedFromAbove, NoTerminator]> {
  let summary = "Define a new dialect";
  let description = [{
    `irdlssa.dialect` operations defines a dialect.

    Example:

    ```mlir
    irdlssa.dialect cmath {
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
// IRDL-SSA Type and Attribute parameters definition
//===----------------------------------------------------------------------===//

def IRDL_SSA_ParametersOp : IRDL_SSA_Op<"parameters", [HasParent<"SSA_TypeOp">]> {
  let summary = "Define the parameters of a type/attribute definition";
  let description = [{
    `irdlssa.parameters` defines new parameters belonging to the type or attribute
    defined by the parent operation.

    Example:

    ```mlir
    irdlssa.dialect cmath {
      irdlssa.type complex {
        %0 = irdlssa.is_type : i32
        %1 = irdlssa.is_type : i64
        %2 = irdlssa.any_of(%0, %1)
        irdlssa.parameters(%2)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath`. The
    type has a single parameter that should be either `i32` or `i64`.
  }];

  let arguments = (ins Variadic<IRDL_SSA_ConstraintType>:$args);
  let assemblyFormat = [{ `(` $args `)` attr-dict }];
}

//===----------------------------------------------------------------------===//
// IRDL-SSA Type definition
//===----------------------------------------------------------------------===//

def IRDL_SSA_TypeOp : IRDL_SSA_Op<"type", [HasParent<"SSA_DialectOp">, NoTerminator, NoRegionArguments, AtMostOneChildOf<"SSA_ParametersOp">]> {
  let summary = "Define a new type";
  let description = [{
    `irdlssa.type` defines new types belonging to the previously defined dialect
    using `irdlssa.dialect`.

    The type parameters can be defined with an `irdlssa.parameters` operation in
    the optional region.

    Example:

    ```mlir
    irdlssa.dialect cmath {
      irdlssa.type complex {
        %0 = irdlssa.is_type : i32
        %1 = irdlssa.is_type : i64
        %2 = irdlssa.any_of(%0, %1)
        irdlssa.parameters(%2)
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
    SSA_DialectOp getDialectOp() { return cast<SSA_DialectOp>(getOperation()->getParentOp()); };
  }];
}

//===----------------------------------------------------------------------===//
// IRDL-SSA Operation definition
//===----------------------------------------------------------------------===//

def IRDL_SSA_OperandsOp : IRDL_SSA_Op<"operands", [HasParent<"SSA_OperationOp">]> {
  let summary = "Define the operands of an operation";
  let description = [{
    `irdlssa.operands` define the operands of the `irdlssa.operation` parent operation
    definition.

    Example:

    ```mlir
    irdlssa.dialect testd {
      irdlssa.operation test_op {
        %0 = irdlssa.is_type : i32
        %1 = irdlssa.is_type : i64
        %2 = irdlssa.any_of(%0, %1)
        irdlssa.operands(%0, %2)
      }
    }
    ```

    The above program defines an operation `test_op` with two expected operands.
    The first operand is expected to be of type `i32`, and the second operand is
    expected to be either of type `i32` or `i64`.
  }];

  let arguments = (ins Variadic<IRDL_SSA_ConstraintType>:$args);
  let assemblyFormat = [{ `(` $args `)` attr-dict }];
}

def IRDL_SSA_ResultsOp : IRDL_SSA_Op<"results", [HasParent<"SSA_OperationOp">]> {
  let summary = "Define the results of an operation";
  let description = [{
    `irdlssa.results` define the results of the `irdlssa.operation` parent operation
    definition.

    Example:

    ```mlir
    irdlssa.dialect testd {
      irdlssa.operation test_op {
        %0 = irdlssa.is_type : i32
        %1 = irdlssa.is_type : i64
        %2 = irdlssa.any_of(%0, %1)
        irdlssa.results(%0, %2)
      }
    }
    ```

    The above program defines an operation `test_op` with two expected results.
    The first result is expected to be an `i32`, and the second result is
    expected to be either an `i32` or an `i64`.
  }];

  let arguments = (ins Variadic<IRDL_SSA_ConstraintType>:$args);
  let assemblyFormat = [{ `(` $args `)` attr-dict }];
}

def IRDL_SSA_OperationOp : IRDL_SSA_Op<"operation",
    [HasParent<"SSA_DialectOp">, NoTerminator, NoRegionArguments,
    AtMostOneChildOf<"SSA_OperandsOp, SSA_ResultsOp">]> {
  let summary = "Define a new operation";
  let description = [{
    `irdlssa.operation` defines new operations belonging to the previously
    defined dialect using `irdlssa.dialect`.

    Those operations have no verifiers.

    Example:

    ```mlir
    irdlssa.dialect "cmath" {
      irdlssa.operation "mul"
    }
    ```
    
    The above program defines an operation `mul` inside the dialect `cmath`.
  }];

  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "custom<KeywordOrString>($name) attr-dict-with-keyword custom<SingleBlockRegion>($body)";

  let extraClassDeclaration = [{
    /// Get the parent dialect operation.
    SSA_DialectOp getDialectOp() { return cast<SSA_DialectOp>(getOperation()->getParentOp()); };
  }];
}

//===----------------------------------------------------------------------===//
// IRDL-SSA Constraint operations
//===----------------------------------------------------------------------===//

class IRDL_SSA_ConstraintOp<string mnemonic, list<Trait> traits = []>
    : IRDL_SSA_Op<mnemonic, [VerifyConstraintInterface, DeclareOpInterfaceMethods<VerifyConstraintInterface>] # traits> {
}

def IRDL_SSA_IsType : IRDL_SSA_ConstraintOp<"is_type", [ParentOneOf<["SSA_TypeOp", "SSA_OperationOp"]>]> {
  let summary = "Constraints to a specific type";
  let description = [{
    `irdlssa.is_type` defines a constraint that only accepts a specific type.

    Example:

    ```mlir
    irdlssa.dialect cmath {
      irdlssa.type complex_restrictive {
        %0 = irdlssa.is_type : i32
        irdlssa.parameters(%0)
      }
    }
    ```

    The above program defines a type `complex_restrictive` inside the dialect `cmath` that
    can only have `i32` as its parameter.
  }];

  let arguments = (ins TypeAttr:$type);
  let results = (outs IRDL_SSA_ConstraintType:$output);
  let assemblyFormat = [{ `:` $type attr-dict }];
}

def IRDL_SSA_ParametricType : IRDL_SSA_ConstraintOp<"parametric_type", [ParentOneOf<["SSA_TypeOp", "SSA_OperationOp"]>]> {
  let summary = "Constraints to a specific instanciated type";
  let description = [{
    `irdlssa.parametric_type` defines a constraint that accepts either a type defined
    in IRDL-SSA or a type wrapper. If the requested type is parametric, it will
    additionally take as operands constraints on its type arguments.

    Example:

    ```mlir
    irdlssa.dialect cmath {
      // ...

      irdlssa.operation norm {
        %0 = irdlssa.any_type
        %1 = irdlssa.parametric_type : "cmath.complex"<%0>
        irdlssa.operands(%1)
        irdlssa.results(%0)
      }
    }
    ```

    The above program defines an operation `norm` inside the dialect `cmath` that
    for any `T` takes a `cmath.complex` with parameter `T` and returns a `T`.
  }];

  let arguments = (ins StrAttr:$type, Variadic<IRDL_SSA_ConstraintType>:$args);
  let results = (outs IRDL_SSA_ConstraintType:$output);
  let assemblyFormat = [{ `:` $type (`<` $args^ `>`)? attr-dict }];
}

def IRDL_SSA_AnyType : IRDL_SSA_ConstraintOp<"any_type", [ParentOneOf<["SSA_TypeOp", "SSA_OperationOp"]>]> {
  let summary = "Constraints to any type";
  let description = [{
    `irdlssa.any_type` defines a constraint that accepts any type.

    Example:

    ```mlir
    irdlssa.dialect cmath {
      irdlssa.type complex_flexible {
        %0 = irdlssa.any_type
        irdlssa.parameters(%0)
      }
    }
    ```

    The above program defines a type `complex_flexible` inside the dialect `cmath` that
    has one type parameter that can be any type.
  }];

  let results = (outs IRDL_SSA_ConstraintType:$output);
  let assemblyFormat = [{ attr-dict }];
}

def IRDL_SSA_AnyOf : IRDL_SSA_ConstraintOp<"any_of", [ParentOneOf<["SSA_TypeOp", "SSA_OperationOp"]>]> {
  let summary = "Constraints to any of the provided types";
  let description = [{
    `irdlssa.is_type` defines a constraint that accepts any type that
    satisfies at least one of its provided type constraints.

    Example:

    ```mlir
    irdlssa.dialect cmath {
      irdlssa.type complex {
        %0 = irdlssa.is_type : i32
        %1 = irdlssa.is_type : i64
        %2 = irdlssa.is_type : f32
        %3 = irdlssa.is_type : f64
        %4 = irdlssa.any_of(%0, %1, %2, %3)
        irdlssa.parameters(%4)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath` that
    can has one type parameter that can be either `i32`, `i64`, `f32` or `f32`.
  }];

  let arguments = (ins Variadic<IRDL_SSA_ConstraintType>:$args);
  let results = (outs IRDL_SSA_ConstraintType:$output);
  let assemblyFormat = [{ `(` $args `)` attr-dict }];
}

#endif // DYN_DIALECT_IRDL_SSA_IR_IRDLSSAOPS
