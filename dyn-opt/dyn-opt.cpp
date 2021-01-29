//===- dyn-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynamicType.h"
#include "MlirOptMain.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "Dyn/DynamicOperation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace dyn;

/// Register a dialect and returns it.
/// Assert in case of error.
DynamicDialect *registerDialect(DynamicContext &ctx, StringRef name) {
  auto dynCtxRes = ctx.createAndRegisterDialect(name);
  if (failed(dynCtxRes)) {
    llvm::errs() << "Failed while registering dialect " << name << "\n";
    abort();
  }
  return *dynCtxRes;
}

/// Register a type in a dialect.
/// Assert in case of error.
DynamicTypeDefinition *registerType(DynamicDialect *dialect, StringRef name) {
  auto typeRes = dialect->createAndAddType(name);
  if (failed(typeRes)) {
    llvm::errs() << "Failed while registering type " << name << " in dialect "
                 << dialect->getName() << "\n";
    abort();
  }
  return *typeRes;
}

/// Register an operation in a dialect.
/// Assert in case of error.
void registerOperation(
    DynamicDialect *dialect, StringRef name,
    std::vector<std::function<LogicalResult(Operation *)>> verifiers = {}) {
  auto registerFailed = failed(dialect->createAndAddOperation(name, verifiers));
  if (registerFailed) {
    llvm::errs() << "Failed while registering operation " << name
                 << " in dialect " << dialect->getName() << "\n";
    abort();
  }
}

/// Register the dyn dialect, used in tests.
void registerDyn(DynamicContext &ctx) {
  auto *dialect = registerDialect(ctx, "dyn");
  registerType(dialect, "dyntype");
  registerOperation(dialect, "foo");
  registerOperation(dialect, "bar");
}

template <int N> LogicalResult hasNRegions(Operation *op) {
  if (op->getNumRegions() == N)
    return success();
  return op->emitOpError("should have exactly " + std::to_string(N) +
                         " regions");
}

template <int N> LogicalResult hasNResults(Operation *op) {
  if (op->getNumResults() == N)
    return success();
  return op->emitOpError("should have exactly " + std::to_string(N) +
                         " results");
}

template <int N> LogicalResult hasNOperands(Operation *op) {
  if (op->getNumOperands() == N)
    return success();
  return op->emitOpError("should have exactly " + std::to_string(N) +
                         " operands");
}

LogicalResult operandsHaveType(DynamicTypeDefinition *type, Operation *op) {
  for (auto operand : op->getOperands())
    if (!DynamicType::isa(operand.getType(), type))
      return op->emitOpError("should have all operands of type " + type->name);
  return success();
}

LogicalResult resultsHaveType(DynamicTypeDefinition *type, Operation *op) {
  for (auto result : op->getResults())
    if (!DynamicType::isa(result.getType(), type))
      return op->emitOpError("should have all results of type " + type->name);
  return success();
}

/// Register the complex example.
void registerComplex(DynamicContext &ctx) {
  auto *dialect = registerDialect(ctx, "complex");
  auto *realType = registerType(dialect, "real");
  auto *complexType = registerType(dialect, "complex");

  auto operandsAreReal = [=](Operation *op) {
    return operandsHaveType(realType, op);
  };
  auto operandsAreComplex = [=](Operation *op) {
    return operandsHaveType(complexType, op);
  };
  auto resultsAreReal = [=](Operation *op) {
    return resultsHaveType(realType, op);
  };
  auto resultsAreComplex = [=](Operation *op) {
    return resultsHaveType(complexType, op);
  };

  registerOperation(dialect, "make_complex",
                    {hasNRegions<0>, hasNResults<1>, hasNOperands<2>,
                     operandsAreReal, resultsAreComplex});
  registerOperation(dialect, "components",
                    {hasNRegions<0>, hasNResults<2>, hasNOperands<1>,
                     operandsAreComplex, resultsAreReal});
  registerOperation(dialect, "mul",
                    {hasNRegions<0>, hasNResults<1>, hasNOperands<2>,
                     operandsAreComplex, resultsAreComplex});
  registerOperation(dialect, "norm",
                    {hasNRegions<0>, hasNResults<1>, hasNOperands<1>,
                     operandsAreComplex, resultsAreReal});
}

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // TODO: Register dyn passes here.

  MLIRContext ctx;
  DynamicContext dynCtx(&ctx);

  registerDyn(dynCtx);
  registerComplex(dynCtx);

  DialectRegistry &registry = ctx.getDialectRegistry();
  registry.insert<StandardOpsDialect>();

  return failed(mlir::MlirOptMain(argc, argv, "Dyn optimizer driver\n", ctx));
}
