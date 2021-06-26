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
            emitError, params[i].cast<TypeAttr>().getValue())))
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

  dialect->addDynamicType(std::move(type));
}
} // namespace irdl
} // namespace mlir

namespace {
/// Objects representing the type constraints of dynamic operations.
/// Each operand and each result have a name, and a type constraint.
using NamedTypeConstraint =
    std::pair<std::string, std::unique_ptr<TypeConstraint>>;
using OpDefConstraints = std::pair<std::vector<NamedTypeConstraint>,
                                   std::vector<NamedTypeConstraint>>;

LogicalResult verifyOpDefConstraints(Operation *op,
                                     const OpDefConstraints &typeConstrs) {
  auto &operandConstrs = typeConstrs.first;
  auto &resultConstrs = typeConstrs.second;

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

  /// Check that all operands satisfy the constraints.
  for (unsigned i = 0; i < numOperands; ++i) {
    auto operandType = op->getOperand(i).getType();
    auto &constraint = operandConstrs[i].second;
    if (failed(constraint->verifyType(emitError, operandType)))
      return failure();
  }

  /// Check that all results satisfy the constraints.
  for (unsigned i = 0; i < numResults; ++i) {
    auto resultType = op->getResult(i).getType();
    auto &constraint = resultConstrs[i].second;
    if (failed(constraint->verifyType(emitError, resultType)))
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
  OpDefConstraints constraints;

  // Add the operand constraints to the type constraints.
  for (auto &def : opDef.operandDef) {
    auto name = def.first;
    auto constraint =
        def.second.cast<TypeConstraintAttrInterface>().getTypeConstraint();
    constraints.first.emplace_back(name, std::move(constraint));
  }

  // Add the result constraints to the type constraints.
  for (auto &def : opDef.resultDef) {
    auto name = def.first;
    auto constraint =
        def.second.cast<TypeConstraintAttrInterface>().getTypeConstraint();
    constraints.second.emplace_back(name, std::move(constraint));
  }

  // TODO define custom parsers and printers.
  // For now, we can only parse with the operation quote syntax.
  auto parser = [](OpAsmParser &parser, OperationState &result) {
    return failure();
  };
  auto printer = [](Operation *op, OpAsmPrinter &printer) {
    printer.printGenericOp(op);
  };

  auto verifier = [constraints{std::move(constraints)}](Operation *op) {
    return verifyOpDefConstraints(op, constraints);
  };

  auto op = DynamicOpDefinition::get(name, dialect, std::move(verifier),
                                     std::move(parser), std::move(printer));
  for (auto trait : opDef.traitDefs) {
    op->addTrait(trait.second);
  }
  dialect->addDynamicOp(std::move(op));
}
} // namespace irdl
} // namespace mlir
