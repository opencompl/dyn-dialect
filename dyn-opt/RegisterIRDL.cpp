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
#include "LowerIRDL.h"
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSA.h"
#include "Dyn/Dialect/IRDL-SSA/IRDLSSARegistration.h"
#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace mlir;

LogicalResult mlir::registerIRDL(StringRef irdlFile, MLIRContext *ctx) {
  DialectRegistry registry;
  registry.insert<irdl::IRDLDialect>();
  registry.insert<irdlssa::IRDLSSADialect>();
  ctx->appendDialectRegistry(registry);

  auto irdlssa = ctx->getOrLoadDialect<irdlssa::IRDLSSADialect>();

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

  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, ctx);

  // Disable multi-threading when parsing the input file. This removes the
  // unnecessary/costly context synchronization when parsing.
  bool wasThreadingEnabled = ctx->isMultithreadingEnabled();
  ctx->disableMultithreading();

  // Parse the input file.
  auto module(parseSourceFile<ModuleOp>(sourceMgr, ctx));
  
  // Translate to IRDL-SSA.
  PassManager pm(ctx);
  pm.addPass(std::make_unique<irdl::LowerIRDL>(irdlssa->irdlssaContext));
  if (failed(pm.run(*module))) {
    return failure();
  }

  // Register IRDL-SSA dialects.
  LogicalResult registrationResult = irdlssa::registerDialects(module.get());
  ctx->enableMultithreading(wasThreadingEnabled);

  if (failed(registrationResult)) {
    return failure();
  }

  return failure(!module);
}

LogicalResult mlir::registerIRDLSSA(StringRef irdlssaFile, MLIRContext *ctx) {
  DialectRegistry registry;
  registry.insert<irdlssa::IRDLSSADialect>();
  ctx->appendDialectRegistry(registry);

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(irdlssaFile, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Give the buffer to the source manager.
  // This will be picked up by the parser.
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());

  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, ctx);

  // Disable multi-threading when parsing the input file. This removes the
  // unnecessary/costly context synchronization when parsing.
  bool wasThreadingEnabled = ctx->isMultithreadingEnabled();
  ctx->disableMultithreading();

  // Parse the input file and reset the context threading state.
  auto module(parseSourceFile<ModuleOp>(sourceMgr, ctx));
  LogicalResult registrationResult = irdlssa::registerDialects(module.get());
  ctx->enableMultithreading(wasThreadingEnabled);

  if (failed(registrationResult)) {
    return failure();
  }

  return failure(!module);
}
