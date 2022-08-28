//===- IRDLSSARegistration.cpp - IRDL-SSA dialect registration ---- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the registration of MLIR objects from IRDL-SSA operations.
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-Eval/IR/IRDLEval.h"
#include "Dyn/Dialect/IRDL-Eval/IRDLEvalInterpreter.h"
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSAInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"

using namespace mlir;
using namespace irdlssa;

namespace mlir {
namespace irdlssa {

namespace {
// Verifier used for dynamic types.
LogicalResult
irdlssaTypeVerifier(function_ref<InFlightDiagnostic()> emitError,
                    ArrayRef<Attribute> params,
                    ArrayRef<std::unique_ptr<TypeConstraint>> constraints,
                    ArrayRef<size_t> paramConstraints) {
  if (params.size() != paramConstraints.size()) {
    emitError().append("expected ", paramConstraints.size(),
                       " type arguments, but had ", params.size());
    return failure();
  }

  ConstraintVerifier verifier(constraints);

  for (size_t i = 0; i < params.size(); i++) {
    if (!params[i].isa<TypeAttr>()) {
      emitError().append(
          "only type attribute type parameters are currently supported");
      return failure();
    }

    if (failed(verifier.verifyType(emitError,
                                   params[i].cast<TypeAttr>().getValue(),
                                   paramConstraints[i]))) {
      return failure();
    }
  }
  return success();
}
} // namespace

} // namespace irdlssa
} // namespace mlir

namespace {

LogicalResult verifyOpDefConstraints(
    Operation *op, ArrayRef<std::unique_ptr<TypeConstraint>> constraints,
    ArrayRef<size_t> operandConstrs, ArrayRef<size_t> resultConstrs) {
  /// Check that we have the right number of operands.
  auto numOperands = op->getNumOperands();
  auto numExpectedOperands = operandConstrs.size();
  if (numOperands != numExpectedOperands)
    return op->emitOpError(std::to_string(numExpectedOperands) +
                           " operands expected, but got " +
                           std::to_string(numOperands));

  /// Check that we have the right number of results.
  auto numResults = op->getNumResults();
  auto numExpectedResults = resultConstrs.size();
  if (numResults != numExpectedResults)
    return op->emitOpError(std::to_string(numExpectedResults) +
                           " results expected, but got " +
                           std::to_string(numResults));

  auto emitError = [op]() { return op->emitError(); };

  ConstraintVerifier verifier(constraints);

  /// Check that all operands satisfy the constraints.
  for (unsigned i = 0; i < numOperands; ++i) {
    auto operandType = op->getOperand(i).getType();
    if (failed(
            verifier.verifyType({emitError}, operandType, operandConstrs[i]))) {
      return failure();
    }
  }

  /// Check that all results satisfy the constraints.
  for (unsigned i = 0; i < numResults; ++i) {
    auto resultType = op->getResult(i).getType();
    if (failed(
            verifier.verifyType({emitError}, resultType, resultConstrs[i]))) {
      return failure();
    }
  }

  return success();
}
} // namespace

namespace mlir {
namespace irdlssa {
/// Register an operation represented by a `irdl.operation` operation.
WalkResult registerOperation(ExtensibleDialect *dialect, SSA_OperationOp op) {
  // If an IRDL-Eval verifier is registered, use it.
  for (Operation &childOp : op.getOps()) {
    using irdleval::Verifier;
    if (Verifier opVerifier = llvm::dyn_cast<Verifier>(childOp)) {
      auto interpreter = irdleval::IRDLEvalInterpreter::compile(
          [&]() { return opVerifier.emitError(); }, opVerifier.getContext(),
          opVerifier);

      if (!interpreter.has_value()) {
        return WalkResult::interrupt();
      }

      size_t numExpectedResults = 0;
      auto resultsOp = op.getOp<SSA_ResultsOp>();
      if (resultsOp.has_value()) {
        numExpectedResults = resultsOp->args().size();
      }

      size_t numExpectedOperands = 0;
      auto operandsOp = op.getOp<SSA_OperandsOp>();
      if (operandsOp.has_value()) {
        numExpectedOperands = operandsOp->args().size();
      }

      auto verifier = [interpreter(std::move(interpreter)), numExpectedResults,
                       numExpectedOperands](Operation *op) -> LogicalResult {
        /// Check that we have the right number of operands.
        size_t numOperands = op->getNumOperands();
        if (numOperands != numExpectedOperands)
          return op->emitOpError(std::to_string(numExpectedOperands) +
                                 " operands expected, but got " +
                                 std::to_string(numOperands));

        /// Check that we have the right number of results.
        size_t numResults = op->getNumResults();
        if (numResults != numExpectedResults)
          return op->emitOpError(std::to_string(numExpectedResults) +
                                 " results expected, but got " +
                                 std::to_string(numResults));

        SmallVector<Type> args;
        for (Value operand : op->getOperands()) {
          args.push_back(operand.getType());
        }
        for (Value result : op->getResults()) {
          args.push_back(result.getType());
        }

        return interpreter->getVerifier().verify(
            [&]() { return op->emitError(); }, args);
      };

      auto parser = [](OpAsmParser &parser, OperationState &result) {
        return failure();
      };
      auto printer = [](Operation *op, OpAsmPrinter &printer, StringRef) {
        printer.printGenericOp(op);
      };

      auto regionVerifier = [](Operation *op) { return success(); };

      auto opDef = DynamicOpDefinition::get(
          op.name(), dialect, std::move(verifier), std::move(regionVerifier),
          std::move(parser), std::move(printer));
      dialect->registerDynamicOp(std::move(opDef));

      return WalkResult::advance();
    }
  }

  // If no IRDL-Eval verifier is registered, fall back to the dynamic
  // evaluation.

  // Resolve SSA values to verifier constraint slots
  SmallVector<Value> constrToValue;
  for (auto &op : op->getRegion(0).getOps()) {
    if (llvm::isa<VerifyConstraintInterface>(op)) {
      assert(op.getNumResults() == 1 &&
             "IRDL-SSA constraint operations must have exactly one result");
      constrToValue.push_back(op.getResult(0));
    }
  }

  // Build the verifiers for each constraint slot
  SmallVector<std::unique_ptr<TypeConstraint>> constraints;
  for (Value v : constrToValue) {
    VerifyConstraintInterface op =
        llvm::cast<VerifyConstraintInterface>(v.getDefiningOp());
    auto verifier = op.getVerifier(constrToValue);
    if (!verifier.has_value()) {
      return WalkResult::interrupt();
    }
    constraints.push_back(std::move(*verifier));
  }

  SmallVector<size_t> operandConstraints;
  SmallVector<size_t> resultConstraints;

  // Gather which constraint slots correspond to operand constraints
  auto operandsOp = op.getOp<SSA_OperandsOp>();
  if (operandsOp.has_value()) {
    operandConstraints.reserve(operandsOp->args().size());
    for (auto operand : operandsOp->args()) {
      for (size_t i = 0; i < constrToValue.size(); i++) {
        if (constrToValue[i] == operand) {
          operandConstraints.push_back(i);
          break;
        }
      }
    }
  }

  // Gather which constraint slots correspond to result constraints
  auto resultsOp = op.getOp<SSA_ResultsOp>();
  if (resultsOp.has_value()) {
    resultConstraints.reserve(resultsOp->args().size());
    for (auto result : resultsOp->args()) {
      for (size_t i = 0; i < constrToValue.size(); i++) {
        if (constrToValue[i] == result) {
          resultConstraints.push_back(i);
          break;
        }
      }
    }
  }

  auto parser = [](OpAsmParser &parser, OperationState &result) {
    return failure();
  };
  auto printer = [](Operation *op, OpAsmPrinter &printer, StringRef) {
    printer.printGenericOp(op);
  };

  auto verifier =
      [constraints{std::move(constraints)},
       operandConstraints{std::move(operandConstraints)},
       resultConstraints{std::move(resultConstraints)}](Operation *op) {
        return verifyOpDefConstraints(op, constraints, operandConstraints,
                                      resultConstraints);
      };

  auto regionVerifier = [](Operation *op) { return success(); };

  auto opDef = DynamicOpDefinition::get(op.name(), dialect, std::move(verifier),
                                        std::move(regionVerifier),
                                        std::move(parser), std::move(printer));
  dialect->registerDynamicOp(std::move(opDef));

  return WalkResult::advance();
}
} // namespace irdlssa
} // namespace mlir

static WalkResult registerType(ExtensibleDialect *dialect, SSA_TypeOp op) {
  // If an IRDL-Eval verifier is registered, use it.
  for (Operation &childOp : op.getOps()) {
    using irdleval::Verifier;
    if (Verifier opVerifier = llvm::dyn_cast<Verifier>(childOp)) {
      auto interpreter = irdleval::IRDLEvalInterpreter::compile(
          [&]() { return opVerifier.emitError(); }, opVerifier.getContext(),
          opVerifier);

      if (!interpreter.has_value()) {
        return WalkResult::interrupt();
      }

      auto verifier = [interpreter(std::move(interpreter))](
                          function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<Attribute> params) -> LogicalResult {
        SmallVector<Type> args;
        for (Attribute attr : params) {
          if (TypeAttr typeAttr = attr.dyn_cast<TypeAttr>()) {
            args.push_back(typeAttr.getValue());
          } else {
            return emitError().append("only type attribute type parameters are "
                                      "currently supported, got ",
                                      attr);
          }
        }
        return interpreter->getVerifier().verify(emitError, args);
      };

      auto type =
          DynamicTypeDefinition::get(op.name(), dialect, std::move(verifier));

      dialect->registerDynamicType(std::move(type));

      return WalkResult::advance();
    }
  }

  // If no IRDL-Eval verifier is registered, fall back to the dynamic
  // evaluation.

  // Resolve SSA values to verifier constraint slots
  SmallVector<Value> constrToValue;
  for (auto &op : op->getRegion(0).getOps()) {
    if (llvm::isa<VerifyConstraintInterface>(op)) {
      assert(op.getNumResults() == 1 &&
             "IRDL-SSA constraint operations must have exactly one result");
      constrToValue.push_back(op.getResult(0));
    }
  }

  // Build the verifiers for each constraint slot
  SmallVector<std::unique_ptr<TypeConstraint>> constraints;
  for (Value v : constrToValue) {
    VerifyConstraintInterface op =
        llvm::cast<VerifyConstraintInterface>(v.getDefiningOp());
    auto verifier = op.getVerifier(constrToValue);
    if (!verifier.has_value()) {
      return WalkResult::interrupt();
    }
    constraints.push_back(std::move(*verifier));
  }

  // Gather which constraint slots correspond to parameter constraints
  auto params = op.getOp<SSA_ParametersOp>();
  SmallVector<size_t> paramConstraints;
  if (params.has_value()) {
    paramConstraints.reserve(params->args().size());
    for (auto param : params->args()) {
      for (size_t i = 0; i < constrToValue.size(); i++) {
        if (constrToValue[i] == param) {
          paramConstraints.push_back(i);
          break;
        }
      }
    }
  }

  auto verifier = [paramConstraints{std::move(paramConstraints)},
                   constraints{std::move(constraints)}](
                      function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<Attribute> params) {
    return irdlssaTypeVerifier(emitError, params, constraints,
                               paramConstraints);
  };

  auto type =
      DynamicTypeDefinition::get(op.name(), dialect, std::move(verifier));

  dialect->registerDynamicType(std::move(type));

  return WalkResult::advance();
}

static WalkResult registerDialect(SSA_DialectOp op) {
  auto *ctx = op.getContext();
  auto dialectName = op.name();

  ctx->getOrLoadDynamicDialect(dialectName, [](DynamicDialect *dialect) {});

  auto *dialect =
      llvm::dyn_cast<ExtensibleDialect>(ctx->getLoadedDialect(dialectName));
  assert(dialect && "extensible dialect should have been registered.");

  WalkResult res =
      op.walk([&](SSA_TypeOp op) { return registerType(dialect, op); });
  if (res.wasInterrupted())
    return res;

  return op.walk(
      [&](SSA_OperationOp op) { return registerOperation(dialect, op); });
}

namespace mlir {
namespace irdlssa {
LogicalResult registerDialects(ModuleOp op) {
  WalkResult res =
      op.walk([&](SSA_DialectOp dialect) { return registerDialect(dialect); });
  return failure(res.wasInterrupted());
}
} // namespace irdlssa
} // namespace mlir
