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
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.h"
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

cl::opt<bool> emitOnlyAny("emit-only-any",
                          cl::desc("Emit only Any constraints"),
                          cl::init(false));

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

Optional<StringRef> cppToIRDLTypeName(StringRef cppName) {
  if (cppName == "::mlir::shape::SizeType")
    return {"shape.size"};
  if (cppName == "::mlir::shape::ShapeType")
    return {"shape.shape"};
  if (cppName == "::mlir::IndexType")
    return {"builtin.index"};
  if (cppName == "::mlir::TensorType")
    return {"builtin.tensor"};
  if (cppName == "::mlir::VectorType")
    return {"builtin.vector"};
  return {};
}

Attribute extractConstraint(MLIRContext *ctx, tblgen::Pred predTblgen) {
  if (emitOnlyAny)
    return AnyTypeConstraintAttr::get(ctx);
  const Record &predRec = predTblgen.getDef();
  auto predStr = predTblgen.getCondition();
  auto pred = removeOuterParentheses(predStr).trim();

  // Any constraint
  if (pred == "true")
    return AnyTypeConstraintAttr::get(ctx);

  // AnyOf constraint
  if (predRec.isSubClassOf("Or")) {
    std::vector<Attribute> constraints;
    for (auto *child : predRec.getValueAsListOfDefs("children")) {
      constraints.push_back(extractConstraint(ctx, tblgen::Pred(child)));
    }
    return AnyOfTypeConstraintAttr::get(ctx, constraints);
  }

  // And constraint
  if (predRec.isSubClassOf("And")) {
    std::vector<Attribute> constraints;
    for (auto *child : predRec.getValueAsListOfDefs("children")) {
      constraints.push_back(extractConstraint(ctx, tblgen::Pred(child)));
    }
    return AndTypeConstraintAttr::get(ctx, constraints);
  }

  // BaseCppClass
  // TODO: change this to a TypeWrapperBaseConstraint
  if (pred.startswith("$_self.isa<") && pred.endswith(">()")) {
    if (auto irdlName = cppToIRDLTypeName(pred.slice(11, pred.size() - 3)))
      return DynTypeBaseConstraintAttr::get(ctx, *irdlName);
  }

  // FloatType constraint
  if (pred == ("$_self.isa<::mlir::FloatType>()")) {
    std::vector<Attribute> constraints;
    constraints.push_back(DynTypeBaseConstraintAttr::get(ctx, "builtin.bf16"));
    constraints.push_back(DynTypeBaseConstraintAttr::get(ctx, "builtin.f16"));
    constraints.push_back(DynTypeBaseConstraintAttr::get(ctx, "builtin.f32"));
    constraints.push_back(DynTypeBaseConstraintAttr::get(ctx, "builtin.f64"));
    constraints.push_back(DynTypeBaseConstraintAttr::get(ctx, "builtin.f80"));
    constraints.push_back(DynTypeBaseConstraintAttr::get(ctx, "builtin.f128"));
    return AnyOfTypeConstraintAttr::get(ctx, constraints);
  }

  llvm::errs() << "Cannot resolve constraint: " << pred << "\n";
  llvm::errs() << predRec << "\n\n";
  return AnyTypeConstraintAttr::get(ctx);
}

Attribute extractConstraint(MLIRContext *ctx,
                            const tblgen::Constraint &constraint) {
  return extractConstraint(ctx, constraint.getPredicate());
}

/// Extract an operation to IRDL.
void extractOperation(OpBuilder &builder, tblgen::Operator &tblgenOp,
                      RecordKeeper &records) {
  auto ctx = builder.getContext();
  auto dialectName = tblgenOp.getDialectName();
  auto opName = tblgenOp.getOperationName();

  // Remove the dialect name from the operation name.
  // We first check that the dialect name is a prefix of the operation name,
  // which is not the case for some operations.
  if (((StringRef)opName).startswith(dialectName))
    opName = std::string(opName.begin() + dialectName.size() + 1, opName.end());

  auto op = builder.create<irdl::OperationOp>(UnknownLoc::get(ctx),
                                              StringAttr::get(ctx, opName));

  // Add the block in the region
  auto &opBlock = op.getBody().emplaceBlock();
  builder.setInsertionPoint(&opBlock, opBlock.begin());

  // Extract operands
  SmallVector<Attribute> operands;
  for (auto &tblgenOperand : tblgenOp.getOperands()) {
    auto constraint = extractConstraint(ctx, tblgenOperand.constraint);
    auto operand =
        NamedTypeConstraintAttr::get(ctx, tblgenOperand.name, constraint);
    operands.push_back(operand);
  }
  auto irdlOperands = ArrayAttr::get(ctx, operands);
  builder.create<OperandsOp>(UnknownLoc::get(ctx), irdlOperands);

  // Extract results
  SmallVector<Attribute> results;
  for (auto &tblgenResult : tblgenOp.getResults()) {
    auto constraint = extractConstraint(ctx, tblgenResult.constraint);
    auto result =
        NamedTypeConstraintAttr::get(ctx, tblgenResult.name, constraint);
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
  auto &dialectBlock = dialect.getBody().emplaceBlock();
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
