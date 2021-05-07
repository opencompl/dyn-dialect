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
#include "Dyn/DynamicType.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"

using namespace mlir;
using namespace irdl;

namespace mlir {
namespace irdl {
LogicalResult registerType(dyn::DynamicDialect *dialect, StringRef name) {
  return dialect->getDynamicContext()->createAndRegisterType(
      name, dialect,
      [name = std::string(name)](function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<Attribute> params) {
        if (params.empty()) {
          return success();
        }
        return LogicalResult(
            emitError().append("Type ", name, " does not have parameters"));
      });
}

LogicalResult registerTypeAlias(dyn::DynamicDialect *dialect, StringRef name,
                                Type type) {
  return dialect->getDynamicContext()->addTypeAlias(name, dialect, type);
}
} // namespace irdl
} // namespace mlir

namespace {
/// Objects representing the type constraints of dynamic operations.
/// Each operand and each result have a name, and a type constraint.
using NamedTypeConstraint =
    std::pair<std::string, std::unique_ptr<TypeConstraint>>;
using OpTypeConstraints = std::pair<std::vector<NamedTypeConstraint>,
                                    std::vector<NamedTypeConstraint>>;

LogicalResult verifyOpTypeConstraints(Operation *op,
                                      const OpTypeConstraints &typeConstrs) {
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
    if (failed(constraint->verifyType(op, operandType, true, i)))
      return failure();
  }

  /// Check that all results satisfy the constraints.
  for (unsigned i = 0; i < numResults; ++i) {
    auto resultType = op->getResult(i).getType();
    auto &constraint = resultConstrs[i].second;
    if (failed(constraint->verifyType(op, resultType, true, i)))
      return failure();
  }

  return success();
}

/// Create a verifier given type constraints and trait implementations
AbstractOperation::VerifyInvariantsFn
createVerifier(OpTypeConstraints constraints,
               std::vector<dyn::DynamicOpTrait *> traits) {
  return [traits{std::move(traits)},
          constraints{std::move(constraints)}](Operation *op) {
    if (failed(verifyOpTypeConstraints(op, constraints)))
      return failure();
    return success(llvm::all_of(traits, [op](auto *trait) {
      return succeeded(trait->verifyTrait(op));
    }));
  };
}
} // namespace

namespace mlir {
namespace irdl {
/// Register an operation represented by a `irdl.operation` operation.
LogicalResult registerOperation(dyn::DynamicDialect *dialect, StringRef name,
                                OpTypeDef opTypeDef) {
  OpTypeConstraints constraints;
  auto *ctx = dialect->getDynamicContext();

  // Add the operand constraints to the type constraints.
  for (auto &def : opTypeDef.operandDef) {
    auto name = def.first;
    auto constraint =
        def.second.cast<TypeConstraintAttrInterface>().getTypeConstraint(*ctx);
    constraints.first.emplace_back(name, std::move(constraint));
  }

  // Add the result constraints to the type constraints.
  for (auto &def : opTypeDef.resultDef) {
    auto name = def.first;
    auto constraint =
        def.second.cast<TypeConstraintAttrInterface>().getTypeConstraint(*ctx);
    constraints.second.emplace_back(name, std::move(constraint));
  }

  // Add the interfaces implementations.
  std::vector<std::unique_ptr<dyn::DynamicOpInterfaceImpl>> interfaces;
  for (auto interfaceAttr : opTypeDef.getInterfaceDefinitions())
    interfaces.push_back(interfaceAttr.getInterfaceImpl());

  // TODO define custom parsers and printers.
  // For now, we can only parse with the operation quote syntax.
  auto parser = [](OpAsmParser &parser, OperationState &result) {
    return failure();
  };
  auto printer = [](Operation *op, OpAsmPrinter &printer) {
    printer.printGenericOp(op);
  };

  auto verifier = createVerifier(std::move(constraints), opTypeDef.traitDefs);

  return ctx->createAndRegisterOperation(
      name, dialect, std::move(parser), std::move(printer), std::move(verifier),
      opTypeDef.traitDefs, std::move(interfaces));
}
} // namespace irdl
} // namespace mlir
