//===- dyn-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "MlirOptMain.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace dyn;

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // TODO: Register passes here.

  MLIRContext ctx;
  auto dynCtx = ctx.getOrLoadDialect<DynamicContext>();

  if (failed(dynCtx->createAndRegisterOpTrait<OpTrait::SameTypeOperands>(
          "SameTypeOperands")))
    llvm::errs() << "Failed to register trait";

  // Register the standard dialect and the IRDL dialect in the MLIR context
  DialectRegistry registry;
  registry.insert<StandardOpsDialect, irdl::IRDLDialect>();
  ctx.appendDialectRegistry(registry);

  return failed(
      mlir::MlirOptMain(argc, argv, "Dyn optimizer driver\n", *dynCtx));
}
