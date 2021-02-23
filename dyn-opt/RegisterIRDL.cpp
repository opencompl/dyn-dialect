//===- RegisterIRDL.h - Register dialects defined in IRDL -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Takes a mlir file that defines dialects using IRDL, and register the dialects
// in a DynamicContext.
//
//===----------------------------------------------------------------------===//

#include "RegisterIRDL.h"
#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/IRDLRegistration.h"
#include "Dyn/DynamicContext.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

using namespace llvm;
using namespace mlir;

LogicalResult mlir::registerIRDL(StringRef irdlFile,
                                 dyn::DynamicContext *dynContext) {

  auto *context = dynContext->getMLIRCtx();
  auto &registry = context->getDialectRegistry();
  registry.insert<irdl::IRDLDialect>();

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(irdlFile, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Give the buffer to the source manager.
  // This will be picked up by the parser.
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());

  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, context);

  // Disable multi-threading when parsing the input file. This removes the
  // unnecessary/costly context synchronization when parsing.
  bool wasThreadingEnabled = context->isMultithreadingEnabled();
  context->disableMultithreading();

  // Parse the input file and reset the context threading state.
  OwningModuleRef module(parseSourceFile(sourceMgr, context));
  context->enableMultithreading(wasThreadingEnabled);
  if (!module)
    return failure();

  module.get()->walk(
      [&](irdl::DialectOp op) { irdl::registerDialect(op, dynContext); });

  return success();
}
