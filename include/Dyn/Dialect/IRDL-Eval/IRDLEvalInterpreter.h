//===- IRDLEvalInterpreter.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interpreter for IRDL-Eval, useful for testing purposes.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_EVAL_IRDL_EVAL_INTERPRETER_H_
#define DYN_DIALECT_IRDL_EVAL_IRDL_EVAL_INTERPRETER_H_

#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEval.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace irdleval {

class IRDLEvalInterpreter {
  IRDLEvalInterpreter() {}

public:
  struct ExecutionResult {
    bool completed;
    bool succeeded;

    static ExecutionResult progress() { return {false, false}; }
    static ExecutionResult success() { return {true, true}; }
    static ExecutionResult failure() { return {true, false}; }
  };

  struct InterpreterVerifier {
    IRDLEvalInterpreter const &interpreter;

    size_t currentBlock;
    size_t instructionPointer;
    DenseMap<size_t, mlir::Type> slotTable;
    DenseMap<size_t, mlir::Type> typeVariables;

    LogicalResult verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                         ArrayRef<Type> args);
  };

  struct Instruction {
    virtual ExecutionResult interpret(InterpreterVerifier &verifier) = 0;
  };

  static Optional<IRDLEvalInterpreter>
  compile(llvm::function_ref<InFlightDiagnostic()> emitError, MLIRContext *ctx,
          Verifier op);
  InterpreterVerifier getVerifier() const;

private:
  DenseMap<size_t, SmallVector<std::unique_ptr<Instruction>>> program;
  SmallVector<size_t> argTypeVariables;
};

} // namespace irdleval
} // namespace mlir

#endif // DYN_DIALECT_IRDL_EVAL_IRDL_EVAL_INTERPRETER_H_
