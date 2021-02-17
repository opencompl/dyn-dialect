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
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace irdl;

namespace {
/// Register a type represented by a `irdl.type` operation.
LogicalResult registerType(TypeOp typeOp, dyn::DynamicDialect *dialect) {
  return dialect->createAndAddType(typeOp.name());
}

/// Objects representing the type constraints of dynamic operations.
/// Each operand and each result have a name, and a type constraint.
using NamedTypeConstraint =
    std::pair<std::string, std::unique_ptr<TypeConstraint>>;
using OpTypeConstraints = std::pair<std::vector<NamedTypeConstraint>,
                                    std::vector<NamedTypeConstraint>>;

LogicalResult verifyOpTypeConstraints(Operation *op,
                                      const OpTypeConstraints &typeConstrs,
                                      dyn::DynamicContext &ctx) {
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

  /// Check that all operands satisfy the constraints.
  for (unsigned i = 0; i < numOperands; ++i) {
    auto operandType = op->getOperand(i).getType();
    auto &constraint = operandConstrs[i].second;
    if (failed(constraint->verifyType(op, operandType, true, i, ctx)))
      return failure();
  }

  /// Check that all results satisfy the constraints.
  for (unsigned i = 0; i < numResults; ++i) {
    auto resultType = op->getResult(i).getType();
    auto &constraint = resultConstrs[i].second;
    if (failed(constraint->verifyType(op, resultType, true, i, ctx)))
      return failure();
  }

  return success();
}

/// Register an operation represented by a `irdl.operation` operation.
LogicalResult registerOperation(OperationOp op, dyn::DynamicDialect *dialect) {
  OpTypeDef opDef = op.op_def();
  OpTypeConstraints constraints;
  auto *ctx = dialect->getDynamicContext();

  for (auto &def : opDef.operandDef) {
    auto name = def.first;
    auto constraint =
        def.second.cast<TypeConstraintAttrInterface>().getTypeConstraint(op,
                                                                         *ctx);
    if (failed(constraint))
      return failure();
    constraints.first.emplace_back(name, std::move(*constraint));
  }

  for (auto &def : opDef.resultDef) {
    auto name = def.first;
    auto constraint =
        def.second.cast<TypeConstraintAttrInterface>().getTypeConstraint(op,
                                                                         *ctx);

    if (failed(constraint))
      return failure();
    constraints.second.emplace_back(name, std::move(*constraint));
  }

  auto typeVerifier =
      [constraints{std::make_shared<OpTypeConstraints>(std::move(constraints))},
       ctx](Operation *op) {
        return verifyOpTypeConstraints(op, *constraints, *ctx);
      };

  return dialect->createAndAddOperation(op.name(), {std::move(typeVerifier)});
}
} // namespace

LogicalResult mlir::irdl::registerDialect(DialectOp dialectOp,
                                          dyn::DynamicContext *ctx) {

  // Register the dialect.
  auto dialectRes = ctx->createAndRegisterDialect(dialectOp.name());
  if (failed(dialectRes))
    return failure();
  auto *dialect = *dialectRes;

  // Register all the types first.
  auto failedTypeRegistration = dialectOp.walk([&](TypeOp type) {
    if (failed(registerType(type, dialect)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  // If a type failed to register, return early with an error.
  if (failedTypeRegistration.wasInterrupted())
    return failure();

  // Register the operations.
  auto failedOperationRegistration = dialectOp.walk([&](OperationOp op) {
    if (failed(registerOperation(op, dialect)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  return failure(failedOperationRegistration.wasInterrupted());
}
