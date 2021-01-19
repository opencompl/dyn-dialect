//===- dyn-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Dyn/DynDialect.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "Dyn/DynamicOperation.h"

using namespace mlir;
using namespace dyn;

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // TODO: Register dyn passes here.

  MLIRContext ctx;
  DynamicContext dynCtx(&ctx);

  // Register a dynamic dialect
  auto fooDialectRes = dynCtx.createAndRegisterDialect("dyn");

  // Check that the dialect is defined
  if (failed(fooDialectRes)) {
    return failed(fooDialectRes);
  }

  DialectRegistry &registry = ctx.getDialectRegistry();
  registry.insert<StandardOpsDialect>();

  return failed(
      mlir::MlirOptMain(argc, argv, "Dyn optimizer driver\n", registry));
}
