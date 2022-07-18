//===- IRDLEvalOps.td - IRDL Eval Dialect ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRDL-Eval dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_EVAL_IR_IRDLEVALOPS
#define DYN_DIALECT_IRDL_EVAL_IR_IRDLEVALOPS

include "IRDLEval.td"
include "IRDLEvalTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSAAttributes.td"

class IRDLEval_Op<string mnemonic, list<Trait> traits = []>
    : Op<IRDLEval_Dialect, mnemonic, traits> {
}

def IRDLEval_Verifier : IRDLEval_Op<"verifier", [ParentOneOf<["mlir::irdlssa::SSA_TypeOp", "mlir::irdlssa::SSA_OperationOp"]>]> {
  let summary = "Defines the verification procedure associated to this op";
  let description = [{
    `irdleval.verifier` represents a specific type-checking automata.
    This instruction is to be inserted within `irdlssa.type` or
    `irdlssa.operation` operations. The automata described by
    `irdleval.verifier` must perform verification equivalent to its
    sibling constraints.

    Its region contains the automata description. The entry block takes
    as parameter the types that are to be checked. If control-flow reaches
    an `irdleval.success` operation, all type checking constraints are
    satisfied. If control-flow reaches an `irdleval.failure` operation, the
    whole type checking operation fails.

    In the case of `irdlssa.operation`, the first arguments to the entry
    block are operands, followed by results.

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

  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$body attr-dict";
  let hasVerifier = 1;
}

def IRDLEval_Success : IRDLEval_Op<"success", [Terminator, ReturnLike]> {
  let summary = "Marks the verification as successful";
  let description = [{
    `irdleval.success` marks successful verification for the entire
    verification checking instrance. If it is reached, the current
    verification is considered successful and execution halts.

    Example:

    ```
    irdleval.verifier {
    ^bb0:
      irdleval.success
    }
    ```

    This verifier succeeds immediately.
  }];

  let arguments = (ins);
  let assemblyFormat = [{ attr-dict }];
}

def IRDLEval_Failure : IRDLEval_Op<"failure", [Terminator, ReturnLike]> {
  let summary = "Marks the verification as failed";
  let description = [{
    `irdleval.failure` marks verification failure for the entire
    verification checking instrance. If it is reached, the current
    verification is considered failed and execution halts.

    Example:

    ```
    irdleval.verifier {
    ^bb0:
      irdleval.failure
    }
    ```

    This verifier fails immediately.
  }];

  let arguments = (ins);
  let assemblyFormat = [{ attr-dict }];
}

def IRDLEval_CheckType : IRDLEval_Op<"check_type", [Terminator]> {
  let summary = "Checks whether the provided type is equal to the specified type";
  let description = [{
    `irdleval.check_type` checks whether the
    type contained in the provided variable matches the type
    described by the associated (potentially parametric) type
    attribute.

    The type attribute must represent a concrete type instance,
    and notably cannot represent types parameterized by other
    type variables. For such use case, see `irdleval.check_parametric`.

    Example:

    ```
    irdleval.verifier {
    ^bb0(%arg: !irdleval.eval_type):
      irdleval.check_type(%arg, "cmath.complex"<f32>, ^bb1, ^bb2)
    ^bb1:
      irdleval.success
    ^bb2:
      irdleval.failure
    }
    ```

    This verifier checks that the `%arg` type variable contains
    the type `cmath.complex<f32>`, failing otherwise.
  }];

  let arguments = (ins IRDLEval_EvalType:$typeVar, ParamTypeAttrOrAnyAttr:$expected);
  let successors = (successor AnySuccessor:$success, AnySuccessor:$failure);
  let assemblyFormat = [{ `(` $typeVar `,` $expected `,` $success `,` $failure `)` attr-dict }];
  let hasVerifier = 1;
}

def IRDLEval_CheckParametric : IRDLEval_Op<"check_parametric", [Terminator]> {
  let summary = "Checks whether a type's base type corresponds to the expected type";
  let description = [{
    `irdleval.check_parametric` checks whether the
    type contained in the provided variable matches the parametric
    structure of the type described by the associated base identifier.

    Control-flow jumps:
    - to the `invalidBase` block if the base type does not match, and
    - to the `success` block if the base type matches. The parametric types
      are in that case passed as arguments to the `success` block, which must as
      such have the same amount of arguments as there are type parameters.
    
    Example:

    ```
    irdleval.verifier {
    ^bb0(%arg: !irdleval.eval_type):
      irdleval.check_parametric(%arg, "cmath.complex", ^bb1, ^bb2)
    ^bb1:
      irdleval.success
    ^bb2:
      irdleval.failure
    }
    ```

    This verifier checks that the argument type is an instance of `cmath.complex`,
    but does not check anything with respect to the validity of its type parameters.
  }];

  let arguments = (ins IRDLEval_EvalType:$typeVar, StrAttr:$base);
  let successors = (successor AnySuccessor:$success, AnySuccessor:$invalidBase);
  let assemblyFormat = [{ `(` $typeVar `,` $base `,` $success `,` $invalidBase `)` attr-dict }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IRDL-Eval Slot operations
//===----------------------------------------------------------------------===//

def IRDLEval_MatchType : IRDLEval_Op<"match_type", [Terminator]> {
  let summary = "Checks whether a type matches the type assigned to a constraint identifier";
  let description = [{
    `irdleval.match_type` checks that the provided slots contains
    the same type as what the provided type variable contains.

    Matching against an empty slot is undefined behavior.

    Example:

    ```
    irdleval.verifier {
    ^bb0(%arg0: !irdleval.eval_type, %arg1: !irdleval.eval_type):
      %0 = irdleval.alloca
      irdleval.assign(%0, %arg0)
      irdleval.match_type(%0, %arg1, ^bb1, ^bb2)
    ^bb1:
      irdleval.success
    ^bb2:
      irdleval.failure
    }
    ```

    This verifier verifies `%arg0` and `%arg1` contains the same
    types. It allocates a slot in the slot table, then assigns the type
    in `%arg0` to the slot. Then, the slot type value is compared to
    the type in `%arg1`, failing if they are not equal.
  }];

  let arguments = (ins IRDLEval_SlotType:$slot, IRDLEval_EvalType:$typeVar);
  let successors = (successor AnySuccessor:$success, AnySuccessor:$failure);
  let assemblyFormat = [{ `(` $slot `,` $typeVar `,` $success `,` $failure `)` attr-dict }];
  let hasVerifier = 1;
}

def IRDLEval_AssignType : IRDLEval_Op<"assign"> {
  let summary = "Assigns a type to a constraint identifier";
  let description = [{
    `irdleval.assign` assigns a type to a slot in the slot
    table.

    Example:

    ```
    irdleval.verifier {
    ^bb0(%arg0: !irdleval.eval_type, %arg1: !irdleval.eval_type):
      %0 = irdleval.alloca
      irdleval.assign(%0, %arg0)
      irdleval.match_type(%0, %arg1, ^bb1, ^bb2)
    ^bb1:
      irdleval.success
    ^bb2:
      irdleval.failure
    }
    ```

    This verifier verifies `%arg0` and `%arg1` contains the same
    types. It allocates a slot `%0` in the slot table, then assigns the type
    in `%arg0` to the slot. Then, the slot type value is compared to
    the type in `%arg1`, failing if they are not equal.
  }];

  let arguments = (ins IRDLEval_SlotType:$slot, IRDLEval_EvalType:$typeVar);
  let assemblyFormat = [{ `(` $slot `,` $typeVar `)` attr-dict }];
}

def IRDLEval_ClearType : IRDLEval_Op<"clear"> {
  let summary = "Clears the type associated to a constraint variable";
  let description = [{
    `irdleval.clear` clears the content of a slot. This is notably useful
    for backtracking, to revert constraint type assignments for example.

    Example:
    irdleval.verifier {
    ^bb0(%arg0: !irdleval.eval_type, %arg1: !irdleval.eval_type, %arg2: !irdleval.eval_type):
      %0 = irdleval.alloca
      irdleval.assign(%0, %arg0)
      irdleval.match_type(%0, %arg2, ^bb1, ^bb2)
    ^bb1:
      irdleval.success
    ^bb2:
      irdleval.clear(%0)
      irdleval.assign(%0, %arg1)
      irdleval.match_type(%0, %arg2, ^bb1, ^bb3)
    ^bb3:
      irdleval.failure
    }
    ```

    This verifier checks that `%arg2` contains the same type as `%arg0` or `%arg1`. It does
    so by assigning the type in `%arg0` in slot `%0`, checking slot `%0` against `%arg2`.
    If this fails, it clears slot `%0` and tries again with the type in `%arg1`.
  }];

  let arguments = (ins IRDLEval_SlotType:$slot);
  let assemblyFormat = [{ `(` $slot `)` attr-dict }];
}

def IRDLEval_Alloca : IRDLEval_Op<"alloca"> {
  let summary = "Allocates a slot to store types";
  let description = [{
    `irdleval.alloca` allocates a slot in the slot table for use.

    Example:

    ```
    irdleval.verifier {
    ^bb0(%arg0: !irdleval.eval_type, %arg1: !irdleval.eval_type):
      %0 = irdleval.alloca
      irdleval.assign(%0, %arg0)
      irdleval.match_type(%0, %arg1, ^bb1, ^bb2)
    ^bb1:
      irdleval.success
    ^bb2:
      irdleval.failure
    }
    ```

    This verifier verifies `%arg0` and `%arg1` contains the same
    types. It allocates a slot `%0` in the slot table, then assigns the type
    in `%arg0` to the slot. Then, the slot type value is compared to
    the type in `%arg1`, failing if they are not equal.
  }];

  let arguments = (ins);
  let results = (outs IRDLEval_SlotType:$output);
  let assemblyFormat = [{ attr-dict }];
}

#endif // DYN_DIALECT_IRDL_EVAL_IR_IRDLEVALOPS
