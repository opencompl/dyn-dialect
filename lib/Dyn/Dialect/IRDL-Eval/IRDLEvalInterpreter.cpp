//===- IRDLEvalInterpreter.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-Eval/IRDLEvalInterpreter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/IRDL/TypeWrapper.h"
#include "mlir/IR/ExtensibleDialect.h"

using namespace mlir;
using namespace irdleval;
using namespace irdlssa;
using namespace irdl;
using Instruction = IRDLEvalInterpreter::Instruction;
using ExecutionResult = IRDLEvalInterpreter::ExecutionResult;

struct GotoInstruction : public Instruction {
  ExecutionResult
  interpret(IRDLEvalInterpreter::InterpreterVerifier &verifier) override {
    verifier.currentBlock = gotoBlock;
    verifier.instructionPointer = 0;
    return ExecutionResult::progress();
  }

  GotoInstruction(size_t gotoBlock) : gotoBlock(gotoBlock) {}

  size_t gotoBlock;
};

struct SuccessInstruction : public Instruction {
  ExecutionResult
  interpret(IRDLEvalInterpreter::InterpreterVerifier &verifier) override {
    return ExecutionResult::success();
  }
};

struct FailureInstruction : public Instruction {
  ExecutionResult
  interpret(IRDLEvalInterpreter::InterpreterVerifier &verifier) override {
    return ExecutionResult::failure();
  }
};

struct CheckTypeInstruction : public Instruction {
  ExecutionResult
  interpret(IRDLEvalInterpreter::InterpreterVerifier &verifier) override {
    auto typeVar = verifier.typeVariables.find(toCheck);
    assert(typeVar != verifier.typeVariables.end() &&
           "type variable is not initialized");
    if (typeVar->second == expected) {
      verifier.currentBlock = gotoOnSuccess;
      verifier.instructionPointer = 0;

    } else {
      verifier.currentBlock = gotoOnFailure;
      verifier.instructionPointer = 0;
    }
    return ExecutionResult::progress();
  }

  CheckTypeInstruction(size_t toCheck, Type expected, size_t gotoOnSuccess,
                       size_t gotoOnFailure)
      : toCheck(toCheck), expected(expected), gotoOnSuccess(gotoOnSuccess),
        gotoOnFailure(gotoOnFailure) {}

  size_t toCheck;
  Type expected;
  size_t gotoOnSuccess;
  size_t gotoOnFailure;
};

struct CheckDynParametricInstruction : public Instruction {
  ExecutionResult
  interpret(IRDLEvalInterpreter::InterpreterVerifier &verifier) override {
    auto typeVar = verifier.typeVariables.find(toCheck);
    assert(typeVar != verifier.typeVariables.end() &&
           "type variable is not initialized");

    auto dynType = typeVar->second.dyn_cast<DynamicType>();
    if (dynType && dynType.getTypeDef() == this->expected) {
      auto params = dynType.getParams();
      assert(params.size() == typeVarsToDefine.size() &&
             "type instance and type definition do not agree on type parameter "
             "amount");
      for (size_t i = 0; i < params.size(); i++) {
        assert(params[i].isa<TypeAttr>() &&
               "general attribute type parameters not supported");
        verifier.typeVariables.insert(
            {typeVarsToDefine[i], params[i].cast<TypeAttr>().getValue()});
      }
      verifier.currentBlock = gotoOnSuccess;
      verifier.instructionPointer = 0;
    } else {
      verifier.currentBlock = gotoOnInvalidBase;
      verifier.instructionPointer = 0;
    }

    return ExecutionResult::progress();
  }

  CheckDynParametricInstruction(size_t toCheck, DynamicTypeDefinition *expected,
                                SmallVector<size_t> typeVarsToDefine,
                                size_t gotoOnSuccess, size_t gotoOnInvalidBase)
      : toCheck(toCheck), expected(expected),
        typeVarsToDefine(typeVarsToDefine), gotoOnSuccess(gotoOnSuccess),
        gotoOnInvalidBase(gotoOnInvalidBase) {}

  size_t toCheck;
  DynamicTypeDefinition *expected;
  SmallVector<size_t> typeVarsToDefine;
  size_t gotoOnSuccess;
  size_t gotoOnInvalidBase;
};

struct CheckParametricInstruction : public Instruction {
  ExecutionResult
  interpret(IRDLEvalInterpreter::InterpreterVerifier &verifier) override {
    auto typeVar = verifier.typeVariables.find(toCheck);
    assert(typeVar != verifier.typeVariables.end() &&
           "type variable is not initialized");

    if (this->expected->isCorrectType(typeVar->second)) {
      auto params = this->expected->getParameters(typeVar->second);
      assert(params.size() == typeVarsToDefine.size() &&
             "type instance and type definition do not agree on type parameter "
             "amount");
      for (size_t i = 0; i < params.size(); i++) {
        assert(params[i].isa<TypeAttr>() &&
               "general attribute type parameters not supported");
        verifier.typeVariables.insert(
            {typeVarsToDefine[i], params[i].cast<TypeAttr>().getValue()});
      }
      verifier.currentBlock = gotoOnSuccess;
      verifier.instructionPointer = 0;
    } else {
      verifier.currentBlock = gotoOnInvalidBase;
      verifier.instructionPointer = 0;
    }

    return ExecutionResult::progress();
  }

  CheckParametricInstruction(size_t toCheck, TypeWrapper *expected,
                             SmallVector<size_t> typeVarsToDefine,
                             size_t gotoOnSuccess, size_t gotoOnInvalidBase)
      : toCheck(toCheck), expected(expected),
        typeVarsToDefine(typeVarsToDefine), gotoOnSuccess(gotoOnSuccess),
        gotoOnInvalidBase(gotoOnInvalidBase) {}

  size_t toCheck;
  TypeWrapper *expected;
  SmallVector<size_t> typeVarsToDefine;
  size_t gotoOnSuccess;
  size_t gotoOnInvalidBase;
};

struct MatchTypeInstruction : public Instruction {
  ExecutionResult
  interpret(IRDLEvalInterpreter::InterpreterVerifier &verifier) override {
    auto typeVar = verifier.typeVariables.find(toCheck);
    assert(typeVar != verifier.typeVariables.end() &&
           "type variable is not initialized");

    auto slotType = verifier.slotTable.find(slot);
    assert(slotType != verifier.slotTable.end() && "slot is not initialized");

    if (typeVar->second == slotType->second) {
      verifier.currentBlock = gotoOnSuccess;
      verifier.instructionPointer = 0;
    } else {
      verifier.currentBlock = gotoOnFailure;
      verifier.instructionPointer = 0;
    }

    return ExecutionResult::progress();
  }

  MatchTypeInstruction(size_t toCheck, size_t slot, size_t gotoOnSuccess,
                       size_t gotoOnFailure)
      : toCheck(toCheck), slot(slot), gotoOnSuccess(gotoOnSuccess),
        gotoOnFailure(gotoOnFailure) {}

  size_t toCheck;
  size_t slot;
  size_t gotoOnSuccess;
  size_t gotoOnFailure;
};

struct AssignTypeInstruction : public Instruction {
  ExecutionResult
  interpret(IRDLEvalInterpreter::InterpreterVerifier &verifier) override {
    auto typeVar = verifier.typeVariables.find(toAssign);
    assert(typeVar != verifier.typeVariables.end() &&
           "type variable is not initialized");

    verifier.slotTable.insert({slot, typeVar->second});

    return ExecutionResult::progress();
  }

  AssignTypeInstruction(size_t toAssign, size_t slot)
      : toAssign(toAssign), slot(slot) {}

  size_t toAssign;
  size_t slot;
};

struct ClearTypeInstruction : public Instruction {
  ExecutionResult
  interpret(IRDLEvalInterpreter::InterpreterVerifier &verifier) override {
    verifier.slotTable.erase(slot);
    return ExecutionResult::progress();
  }

  ClearTypeInstruction(size_t slot) : slot(slot) {}

  size_t slot;
};

IRDLEvalInterpreter::InterpreterVerifier
IRDLEvalInterpreter::getVerifier() const {
  DenseMap<size_t, Type> slotTable;
  DenseMap<size_t, Type> typeVariables;
  return IRDLEvalInterpreter::InterpreterVerifier{*this, 0, 0, slotTable,
                                                  typeVariables};
}

Optional<IRDLEvalInterpreter>
IRDLEvalInterpreter::compile(llvm::function_ref<InFlightDiagnostic()> emitError,
                             MLIRContext *ctx, Verifier op) {
  IRDLEvalInterpreter interpreter;

  DenseMap<Value, size_t> slotToId;
  size_t id = 0;
  op.walk([&](Alloca alloca) {
    slotToId.insert({alloca.getResult(), id});
    id++;
  });

  DenseMap<Block *, size_t> blockToId;
  id = 0;
  for (Block &block : op.getRegion().getBlocks()) {
    blockToId.insert({&block, id});
    id++;
  }

  DenseMap<Value, size_t> typeVarToId;
  id = 0;
  for (Block &block : op.getRegion().getBlocks()) {
    for (Value val : block.getArguments()) {
      typeVarToId.insert({val, id});
      id++;
    }
  }

  for (Value arg : op.getRegion().getArguments()) {
    interpreter.argTypeVariables.push_back(typeVarToId[arg]);
  }

  for (Block &block : op.getRegion().getBlocks()) {
    SmallVector<std::unique_ptr<Instruction>> instructions;

    for (Operation &op : block.getOperations()) {
      bool fatal = false;
      TypeSwitch<Operation *>(&op)
          .Case([&](cf::BranchOp op) {
            instructions.push_back(
                std::make_unique<GotoInstruction>(blockToId[op.getDest()]));
          })
          .Case([&](Success op) {
            instructions.push_back(std::make_unique<SuccessInstruction>());
          })
          .Case([&](Failure op) {
            instructions.push_back(std::make_unique<FailureInstruction>());
          })
          .Case([&](CheckType op) {
            Attribute instantiatedParam =
                op.getExpected().instantiateParamType(emitError, *ctx);

            if (!instantiatedParam) {
              emitError().append("invalid attribute ", op.getExpected());
              fatal = true;
              return;
            }

            Type type;
            if (TypeAttr typeAttr = instantiatedParam.dyn_cast<TypeAttr>()) {
              type = typeAttr.getValue();
            } else {
              emitError().append("attribute ", op.getExpected(),
                                 " is not a type");
              fatal = true;
              return;
            }

            instructions.push_back(std::make_unique<CheckTypeInstruction>(
                typeVarToId[op.getTypeVar()], type, blockToId[op.getSuccess()],
                blockToId[op.getFailure()]));
          })
          .Case([&](CheckParametric op) {
            SmallVector<size_t> typeVarsToDefine;
            for (Value arg : op.getSuccess()->getArguments()) {
              typeVarsToDefine.push_back(typeVarToId[arg]);
            }

            if (DynamicTypeDefinition *dynTypeDef =
                    findDynamicType(*ctx, op.getBase())) {
              instructions.push_back(
                  std::make_unique<CheckDynParametricInstruction>(
                      typeVarToId[op.getTypeVar()], dynTypeDef,
                      typeVarsToDefine, blockToId[op.getSuccess()],
                      blockToId[op.getInvalidBase()]));
            } else if (TypeWrapper *typeWrapper =
                           findTypeWrapper(*ctx, op.getBase())) {
              instructions.push_back(
                  std::make_unique<CheckParametricInstruction>(
                      typeVarToId[op.getTypeVar()], typeWrapper,
                      typeVarsToDefine, blockToId[op.getSuccess()],
                      blockToId[op.getInvalidBase()]));

            } else {
              emitError().append("type ", op.getBase(), " not found");
              fatal = true;
            }
          })
          .Case([&](MatchType op) {
            instructions.push_back(std::make_unique<MatchTypeInstruction>(
                typeVarToId[op.getTypeVar()], slotToId[op.getSlot()],
                blockToId[op.getSuccess()], blockToId[op.getFailure()]));
          })
          .Case([&](AssignType op) {
            instructions.push_back(std::make_unique<AssignTypeInstruction>(
                typeVarToId[op.getTypeVar()], slotToId[op.getSlot()]));
          })
          .Case([&](ClearType op) {
            instructions.push_back(
                std::make_unique<ClearTypeInstruction>(slotToId[op.getSlot()]));
          })
          .Case([&](Alloca op) {})
          .Default([&](Operation *op) {
            emitError().append("unsupported instruction ", op, " in verifier");
            fatal = true;
          });

      if (fatal) {
        return {};
      }
    }

    interpreter.program.insert({blockToId[&block], std::move(instructions)});
  }

  return {std::move(interpreter)};
}

LogicalResult IRDLEvalInterpreter::InterpreterVerifier::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, ArrayRef<Type> args) {
  if (args.size() != this->interpreter.argTypeVariables.size()) {
    return emitError().append("invalid amount of types, expected ",
                              this->interpreter.argTypeVariables.size(),
                              ", got ", args.size());
  }

  for (size_t i = 0; i < args.size(); i++) {
    this->typeVariables.insert(
        {this->interpreter.argTypeVariables[i], args[i]});
  }

  ExecutionResult result = ExecutionResult::progress();
  while (!result.completed) {
    Instruction *nextInst = (this->interpreter.program.find(this->currentBlock)
                                 ->second)[this->instructionPointer]
                                .get();
    this->instructionPointer++;
    result = nextInst->interpret(*this);
  }

  if (!result.succeeded) {
    emitError().append(
        "the provided types do not satisfy the type constraints");
  }

  return LogicalResult::success(result.succeeded);
}
