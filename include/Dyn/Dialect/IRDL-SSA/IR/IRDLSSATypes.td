//===- IRDLSSATypes.td - IRDL Types ------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types IRDL-SSA uses.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_SSA_IR_IRDLSSATYPES
#define DYN_DIALECT_IRDL_SSA_IR_IRDLSSATYPES

include "mlir/IR/AttrTypeBase.td"
include "IRDLSSA.td"

class IRDL_SSA_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<IRDL_SSA_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def IRDL_SSA_ConstraintType : IRDL_SSA_Type<"Constraint", "constr"> {
  let summary = "Type representing a type constraint";
  let description = [{
    The Constraint type is the type of type constraints when
    used in SSA form.

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

    Here, `%0`, `%1` and `%2` are all of type `Constraint`.
  }];
}

#endif // DYN_DIALECT_IRDL_SSA_IR_IRDLSSATYPES