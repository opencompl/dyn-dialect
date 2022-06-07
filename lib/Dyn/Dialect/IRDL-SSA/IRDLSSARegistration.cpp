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
void registerOperation(LogicalResult &res, ExtensibleDialect *dialect,
                       SSA_OperationOp op) {
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
    if (!verifier.hasValue()) {
      res = failure();
      return;
    }
    constraints.push_back(std::move(*verifier));
  }

  SmallVector<size_t> operandConstraints;
  SmallVector<size_t> resultConstraints;

  // Gather which constraint slots correspond to operand constraints
  auto operandsOp = op.getOp<SSA_OperandsOp>();
  if (operandsOp.hasValue()) {
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
  if (resultsOp.hasValue()) {
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
}
} // namespace irdlssa
} // namespace mlir

static void registerType(LogicalResult &res, ExtensibleDialect *dialect,
                         SSA_TypeOp op) {
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
    if (!verifier.hasValue()) {
      res = failure();
      return;
    }
    constraints.push_back(std::move(*verifier));
  }

  // Gather which constraint slots correspond to parameter constraints
  auto params = op.getOp<SSA_ParametersOp>();
  SmallVector<size_t> paramConstraints;
  if (params.hasValue()) {
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
}

static void registerDialect(LogicalResult &res, SSA_DialectOp op) {
  auto *ctx = op.getContext();
  auto dialectName = op.name();

  ctx->getOrLoadDynamicDialect(dialectName, [](DynamicDialect *dialect) {});

  auto *dialect =
      llvm::dyn_cast<ExtensibleDialect>(ctx->getLoadedDialect(dialectName));
  assert(dialect && "extensible dialect should have been registered.");

  op.walk([&](SSA_TypeOp op) { registerType(res, dialect, op); });
  if (failed(res))
    return;

  op.walk([&](SSA_OperationOp op) { registerOperation(res, dialect, op); });
}

namespace mlir {
namespace irdlssa {
LogicalResult registerDialects(ModuleOp op) {
  LogicalResult res = success();
  op.walk([&](SSA_DialectOp dialect) { registerDialect(res, dialect); });
  return res;
}
} // namespace irdlssa
} // namespace mlir
