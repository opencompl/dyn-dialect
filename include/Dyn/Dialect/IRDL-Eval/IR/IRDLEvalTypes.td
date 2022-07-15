//===- IRDLEvalTypes.td - IRDL-Eval Types ------------------*- tablegen -*-===//
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

#ifndef DYN_DIALECT_IRDL_EVAL_IR_IRDLEVALTYPES
#define DYN_DIALECT_IRDL_EVAL_IR_IRDLEVALTYPES

include "mlir/IR/AttrTypeBase.td"
include "IRDLEval.td"

class IRDLEval_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<IRDLEval_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def IRDLEval_EvalType : IRDLEval_Type<"EvalType", "type"> {
  let summary = "Type representing a type";
  let description = [{
    `irdleval.eval_type` is the type of types when stored
    in SSA values.
  }];
}

def IRDLEval_SlotType : IRDLEval_Type<"Slot", "slot"> {
  let summary = "Type representing a slot to hold a type mapped to a constraint";
  let description = [{
    `irdleval.slot` is the type of slots once allocated with
    `irdleval.alloca`.
  }];
}

#endif // DYN_DIALECT_IRDL_EVAL_IR_IRDLEVALTYPES
