//===- IRDLRegistration.h - IRDL registration -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the registration of MLIR objects from IRDL operations.
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IRDLRegistration.h"
#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
#include "Dyn/Dialect/IRDL/TypeConstraint.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"

using namespace mlir;
using namespace irdl;

namespace mlir {
namespace irdl {

namespace {
// Verifier used for dynamic types.
LogicalResult
irdlTypeVerifier(function_ref<InFlightDiagnostic()> emitError,
                 ArrayRef<Attribute> params,
                 ArrayRef<std::unique_ptr<TypeConstraint>> paramConstraints) {
  if (params.size() != paramConstraints.size()) {
    emitError().append("expected ", paramConstraints.size(),
                       " type arguments, but had ", params.size());
    return failure();
  }

  for (size_t i = 0; i < params.size(); i++) {
    if (failed(paramConstraints[i]->verifyType(
            emitError, params[i].cast<TypeAttr>().getValue(), {}, {})))
      return failure();
  }
  return success();
}
} // namespace

void registerType(ExtensibleDialect *dialect, TypeDef typeDef) {
  SmallVector<std::unique_ptr<TypeConstraint>> paramConstraints;
  for (auto param : typeDef.paramDefs) {
    paramConstraints.push_back(
        param.second.cast<TypeConstraintAttrInterface>().getTypeConstraint());
  }

  auto verifier = [paramConstraints{std::move(paramConstraints)}](
                      function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<Attribute> params) {
    return irdlTypeVerifier(emitError, params, paramConstraints);
  };

  auto type =
      DynamicTypeDefinition::get(typeDef.name, dialect, std::move(verifier));

  dialect->registerDynamicType(std::move(type));
}
} // namespace irdl
} // namespace mlir

namespace {

LogicalResult verifyOpDefConstraints(
    Operation *op, ArrayRef<std::unique_ptr<TypeConstraint>> operandConstrs,
    ArrayRef<std::unique_ptr<TypeConstraint>> resultConstrs,
    ArrayRef<std::unique_ptr<TypeConstraint>> typeConstrVars) {
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
  SmallVector<Type> varAssignments(typeConstrVars.size());

  /// Check that all operands satisfy the constraints.
  for (unsigned i = 0; i < numOperands; ++i) {
    auto operandType = op->getOperand(i).getType();
    auto &constraint = operandConstrs[i];
    if (failed(constraint->verifyType(emitError, operandType, typeConstrVars,
                                      varAssignments)))
      return failure();
  }

  /// Check that all results satisfy the constraints.
  for (unsigned i = 0; i < numResults; ++i) {
    auto resultType = op->getResult(i).getType();
    auto &constraint = resultConstrs[i];
    if (failed(constraint->verifyType(emitError, resultType, typeConstrVars,
                                      varAssignments)))
      return failure();
  }

  return success();
}
} // namespace

namespace mlir {
namespace irdl {
/// Register an operation represented by a `irdl.operation` operation.
void registerOperation(ExtensibleDialect *dialect, StringRef name,
                       OpDef opDef) {
  SmallVector<std::unique_ptr<TypeConstraint>> typeConstrVars;
  SmallVector<std::unique_ptr<TypeConstraint>> operandConstraints;
  SmallVector<std::unique_ptr<TypeConstraint>> resultConstraints;

  typeConstrVars.reserve(opDef.typeConstraintVars.size());
  for (auto typeConstr : opDef.typeConstraintVars) {
    auto constraint = typeConstr.second.cast<TypeConstraintAttrInterface>()
                          .getTypeConstraint();
    typeConstrVars.emplace_back(std::move(constraint));
  }

  // Add the operand constraints to the type constraints.
  operandConstraints.reserve(opDef.operandDef.size());
  for (auto &def : opDef.operandDef) {
    auto constraint =
        def.second.cast<TypeConstraintAttrInterface>().getTypeConstraint();
    operandConstraints.emplace_back(std::move(constraint));
  }

  // Add the result constraints to the type constraints.
  resultConstraints.reserve(opDef.resultDef.size());
  for (auto &def : opDef.resultDef) {
    auto constraint =
        def.second.cast<TypeConstraintAttrInterface>().getTypeConstraint();
    resultConstraints.emplace_back(std::move(constraint));
  }

  auto parser = [](OpAsmParser &parser, OperationState &result) {
    return failure();
  };
  auto printer = [](Operation *op, OpAsmPrinter &printer, StringRef) {
    printer.printGenericOp(op);
  };

  auto verifier = [operandConstraints{std::move(operandConstraints)},
                   resultConstraints{std::move(resultConstraints)},
                   typeConstrVars{std::move(typeConstrVars)}](Operation *op) {
    return verifyOpDefConstraints(op, operandConstraints, resultConstraints,
                                  typeConstrVars);
  };

  auto op = DynamicOpDefinition::get(name, dialect, std::move(verifier),
                                     std::move(parser), std::move(printer));
  dialect->registerDynamicOp(std::move(op));
}
} // namespace irdl
} // namespace mlir
