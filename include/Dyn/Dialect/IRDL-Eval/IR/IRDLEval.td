//===- IRDLEval.td - IRDL-Eval Dialect ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRDL-Eval dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_EVAL_IR_IRDL_EVAL
#define DYN_DIALECT_IRDL_EVAL_IR_IRDL_EVAL

include "mlir/IR/OpBase.td"

def IRDLEval_Dialect : Dialect {
  let summary = "IR Definition Language Eval Dialect";
  let description = [{
      IRDL-Eval is a dialect to represent automata checking type constraints.
      These automata are expressed using concepts close to LLVM IR, so that
      it is easy to lower to assembly.

      Verifying automata can be appended to IRDL-SSA definitions by inserting
      `irdleval.verifier` operations within `irdlssa.type` or
      `irdlssa.operation` operations. The verifier operation itself then
      describes an automata that should be equivalent to checking the sibling
      constraint declarations.

      The automata operate on type variables (SSA values of type
      `irdleval.eval_type`). IRDL-Eval operations are sequential type checking
      primitives. Additionally, IRDL-Eval provides a global type slot table
      that allows mapping type slots (allocated via `irdleval.alloca`) to
      concrete types. See instruction documentation for more details.

      Example:

      ```
      irdlssa.operation add_float {
        %0 = irdlssa.is_type : f32
        %1 = irdlssa.is_type : f64
        %2 = irdlssa.any_of(%0, %1)
        irdlssa.operands(%2, %2)
        irdlssa.results(%2)
        
        irdleval.verifier {
        ^bb0(%oper0: !irdleval.eval_type, %oper1: !irdleval.eval_type, %res: !irdleval.eval_type):
          %3 = irdleval.alloca
          irdleval.check_type(%oper0, f32, ^bb3, ^bb1)
        ^bb1:
          irdleval.check_type(%oper0, f64, ^bb3, ^bb2)
        ^bb2:
          irdleval.failure
        ^bb3:
          irdleval.assign(%3, %oper0)
          irdleval.match_type(%3, %oper1, ^bb4, ^bb2)
        ^bb4:
          irdleval.match_type(%3, %res, ^bb5, ^bb2)
        ^bb5:
          irdleval.success
        }
      }
      ```

      This verifier implementation is equivalent to its sibling constraints:
      it checks that the first operand is either `f32` or `f64`, then ensures
      that the second operand type and the result type are equal to the first
      operand type.
  }];

  let useDefaultTypePrinterParser = 1;

  let name = "irdleval";
  let cppNamespace = "::mlir::irdleval";
}

#endif // DYN_DIALECT_IRDL_EVAL_IR_IRDL_EVAL
