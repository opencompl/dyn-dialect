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

#include "mlir/Dialect/IRDL/IR/IRDL.h"
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

/// Check that parentheses are balanced
bool hasBalancedParentheses(StringRef str) {
  int parenLevel = 0;
  for (auto c : str) {
    if (c == '(') {
      parenLevel += 1;
    } else if (c == ')') {
      parenLevel -= 1;
      if (parenLevel < 0)
        return false;
    }
  }
  return parenLevel == 0;
}

/// Remove outer parentheses if they exists.
StringRef removeOuterParentheses(StringRef str) {
  str = str.trim();
  if (str.front() != '(' || str.back() != ')')
    return str;
  auto simplifiedStr = str.slice(1, str.size() - 1);
  if (hasBalancedParentheses(simplifiedStr))
    return removeOuterParentheses(simplifiedStr);
  return str;
}

class Extractor {
public:
  Extractor(RecordKeeper &records, OpBuilder &rootBuilder)
      : records(records), rootBuilder(rootBuilder){};

  Value extractConstraint(OpBuilder &builder, tblgen::Pred tblgenPred) {
    MLIRContext *ctx = builder.getContext();
    llvm::Record predRec = tblgenPred.getDef();
    std::string predStr = tblgenPred.getCondition();
    llvm::StringRef pred = removeOuterParentheses(predStr).trim();

    // Any constraint
    if (pred == "true") {
      auto op = builder.create<AnyOp>(UnknownLoc::get(ctx));
      return op.getOutput();
    }

    // AnyOf constraint
    if (predRec.isSubClassOf("Or")) {
      std::vector<Value> constraints;
      for (auto *child : predRec.getValueAsListOfDefs("children")) {
        constraints.push_back(extractConstraint(builder, tblgen::Pred(child)));
      }
      auto op = builder.create<AnyOfOp>(UnknownLoc::get(ctx), constraints);
      return op.getOutput();
    }

    // AllOf constraint
    if (predRec.isSubClassOf("And")) {
      std::vector<Value> constraints;
      for (auto *child : predRec.getValueAsListOfDefs("children")) {
        constraints.push_back(extractConstraint(builder, tblgen::Pred(child)));
      }
      auto op = builder.create<AllOfOp>(UnknownLoc::get(ctx), constraints);
      return op.getOutput();
    }

    if (predRec.isSubClassOf("ISA")) {
      StringRef isa = predRec.getValueAsString("isa");
      TypeAttr isOpType = nullptr;

      if (isa == "::mlir::IndexType") {
        isOpType = TypeAttr::get(IndexType::get(ctx));
      }

      if (isOpType != nullptr) {
        auto op = builder.create<IsOp>(UnknownLoc::get(ctx), isOpType);
        return op.getOutput();
      }
    }

    auto op = builder.create<CPredOp>(UnknownLoc::get(ctx),
                                      StringAttr::get(ctx, pred));
    return op.getOutput();
  }

  /// Extract an operation to IRDL.
  void extractOperation(OpBuilder &builder, tblgen::Operator &tblgenOp) {
    auto ctx = builder.getContext();
    auto dialectName = tblgenOp.getDialectName();
    auto opName = tblgenOp.getOperationName();

    // Remove the dialect name from the operation name.
    // We first check that the dialect name is a prefix of the operation name,
    // which is not the case for some operations.
    if (((StringRef)opName).startswith(dialectName))
      opName =
          std::string(opName.begin() + dialectName.size() + 1, opName.end());

    auto op = builder.create<irdl::OperationOp>(UnknownLoc::get(ctx),
                                                StringAttr::get(ctx, opName));

    // Add the block in the region
    auto &opBlock = op.getBody().emplaceBlock();
    auto opBuilder = OpBuilder::atBlockBegin(&opBlock);

    // Extract operands
    SmallVector<Value> operands;
    for (auto &tblgenOperand : tblgenOp.getOperands()) {
      auto operand =
          extractConstraint(opBuilder, tblgenOperand.constraint.getPredicate());
      operands.push_back(operand);
    }

    // Extract results
    SmallVector<Value> results;
    for (auto &tblgenResult : tblgenOp.getResults()) {
      auto result =
          extractConstraint(opBuilder, tblgenResult.constraint.getPredicate());
      results.push_back(result);
    }

    // Create the operands and results operations.
    opBuilder.create<OperandsOp>(UnknownLoc::get(ctx), operands);
    opBuilder.create<ResultsOp>(UnknownLoc::get(ctx), results);
  }

  /// Extract the dialect to IRDL
  void extractDialect() {

    auto ctx = rootBuilder.getContext();
    std::vector<Record *> opDefs = getOpDefinitions(records);

    // Retrieve the dialect name.
    assert(opDefs.size() > 0);
    auto dialectName = tblgen::Operator(opDefs[0]).getDialectName();

    // Create the IDRL dialect operation, and set the insertion point in it.
    auto dialect = rootBuilder.create<irdl::DialectOp>(
        UnknownLoc::get(ctx), StringAttr::get(ctx, dialectName));
    auto &dialectBlock = dialect.getBody().emplaceBlock();

    auto builder = OpBuilder::atBlockBegin(&dialectBlock);

    // Walk all TableGen operations, and create new IRDL operations.
    for (auto rec : opDefs) {
      // Create the operation using the TableGen name.
      auto tblgenOp = tblgen::Operator(rec);
      extractOperation(builder, tblgenOp);
    }
  }

private:
  RecordKeeper &records;
  OpBuilder rootBuilder;
};

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

  Extractor extractor(records, builder);

  extractor.extractDialect();

  module->print(os);

  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  return TableGenMain(argv[0], &MlirTableGenStatsMain);
}
