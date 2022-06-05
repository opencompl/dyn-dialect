//===- tblgen-extract.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Extract IRDL from a TableGen file.
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/GenNameParser.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;
using namespace irdl;

std::vector<Record *> getOpDefinitions(const RecordKeeper &recordKeeper) {
  if (!recordKeeper.getClass("Op"))
    return {};
  return recordKeeper.getAllDerivedDefinitions("Op");
}

std::vector<Record *> getTypeDefinitions(const RecordKeeper &recordKeeper) {
  if (!recordKeeper.getClass("TypeDef"))
    return {};
  return recordKeeper.getAllDerivedDefinitions("TypeDef");
}

std::vector<Record *> getAttrDefinitions(const RecordKeeper &recordKeeper) {
  if (!recordKeeper.getClass("AttrDef"))
    return {};
  return recordKeeper.getAllDerivedDefinitions("AttrDef");
}

/// Extract an operation to IRDL.
void extractOperation(OpBuilder &builder, tblgen::Operator &tblgenOp,
                      RecordKeeper &records) {
  auto ctx = builder.getContext();
  auto op = builder.create<irdl::OperationOp>(
      UnknownLoc::get(ctx), StringAttr::get(ctx, tblgenOp.getOperationName()));

  // Add the block in the region
  auto &opBlock = op.body().emplaceBlock();
  builder.setInsertionPoint(&opBlock, opBlock.begin());

  // Extract operands
  SmallVector<Attribute> operands;
  for (auto &tblgenOperand : tblgenOp.getOperands()) {
    auto operand = NamedTypeConstraintAttr::get(
        ctx, tblgenOperand.name, AnyTypeConstraintAttr::get(ctx));
    operands.push_back(operand);
  }
  auto irdlOperands = ArrayAttr::get(ctx, operands);
  builder.create<OperandsOp>(UnknownLoc::get(ctx), irdlOperands);

  // Extract results
  SmallVector<Attribute> results;
  for (auto &tblgenResult : tblgenOp.getResults()) {
    auto result = NamedTypeConstraintAttr::get(ctx, tblgenResult.name,
                                               AnyTypeConstraintAttr::get(ctx));
    results.push_back(result);
  }
  auto irdlResults = ArrayAttr::get(ctx, results);
  builder.create<ResultsOp>(UnknownLoc::get(ctx), irdlResults);

  // Put the insertion point after the created operation.
  builder.setInsertionPointAfter(op);
  assert(succeeded(op.verify()));
}

/// Extract the dialect to IRDL
void extractDialect(OpBuilder &builder, RecordKeeper &records) {
  auto ctx = builder.getContext();
  std::vector<Record *> opDefs = getOpDefinitions(records);
  std::vector<Record *> typeDefs = getTypeDefinitions(records);
  std::vector<Record *> attrDefs = getAttrDefinitions(records);

  // Retrieve the dialect name.
  assert(opDefs.size() > 0);
  auto dialectName = tblgen::Operator(opDefs[0]).getDialectName();

  // Create the IDRL dialect operation, and set the insertion point in it.
  auto dialect = builder.create<irdl::DialectOp>(
      UnknownLoc::get(ctx), StringAttr::get(ctx, dialectName));
  auto &dialectBlock = dialect.body().emplaceBlock();
  builder.setInsertionPoint(&dialectBlock, dialectBlock.begin());

  // Walk all TableGen operations, and create new IRDL operations.
  for (auto rec : opDefs) {
    // Create the operation using the TableGen name.
    auto tblgenOp = tblgen::Operator(rec);
    extractOperation(builder, tblgenOp, records);
  }
}

bool MlirTableGenStatsMain(raw_ostream &os, RecordKeeper &records) {
  // Create the context, and the main module operation.
  MLIRContext ctx;
  ctx.getOrLoadDialect<IRDLDialect>();
  auto unknownLoc = UnknownLoc::get(&ctx);
  OwningOpRef<ModuleOp> module(ModuleOp::create(unknownLoc));

  // Create the builder, and set its insertion point in the module.
  OpBuilder builder(&ctx);
  auto &moduleBlock = module->getRegion().getBlocks().front();
  builder.setInsertionPoint(&moduleBlock, moduleBlock.begin());

  extractDialect(builder, records);

  module->print(os);

  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  return TableGenMain(argv[0], &MlirTableGenStatsMain);
}