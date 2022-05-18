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

  let arguments = (ins Variadic<IRDL_SSA_ContraintType>:$args);
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
// IRDL-SSA Constraint operations
//===----------------------------------------------------------------------===//

def IRDL_SSA_IsType : IRDL_SSA_Op<"is_type"> { // TODO: Constrain parent
  let summary = "Constraints to a specific type";
  let description = [{
    `irdlssa.is_type` defines a constraint that only accepts an instance
    of a specific type.

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
  let results = (outs IRDL_SSA_ContraintType:$output);
  let assemblyFormat = [{ `:` $type attr-dict }];
}



#endif // DYN_DIALECT_IRDL_SSA_IR_IRDLSSAOPS