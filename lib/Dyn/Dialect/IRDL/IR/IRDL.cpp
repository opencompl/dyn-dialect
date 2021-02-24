//===- IRDL.cpp - IRDL dialect ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
#include "Dyn/Dialect/IRDL/TypeConstraint.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::irdl;

//===----------------------------------------------------------------------===//
// IRDL dialect.
//===----------------------------------------------------------------------===//

void IRDLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dyn/Dialect/IRDL/IR/IRDLOps.cpp.inc"
      >();
  addAttributes<OpTypeDefAttr, EqDynTypeConstraintAttr, EqTypeConstraintAttr>();
}

//===----------------------------------------------------------------------===//
// irdl::DialectOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(DialectOp dialectOp) {
  return success(Dialect::isValidNamespace(dialectOp.name()));
}

static ParseResult parseDialectOp(OpAsmParser &p, OperationState &state) {
  Builder &builder = p.getBuilder();

  // Parse the dialect name.
  StringRef name;
  if (failed(p.parseKeyword(&name)))
    return failure();
  state.addAttribute("name", builder.getStringAttr(name));

  // Parse the dialect body.
  Region *region = state.addRegion();
  if (failed(p.parseRegion(*region)))
    return failure();

  DialectOp::ensureTerminator(*region, builder, state.location);

  return success();
}

static void print(OpAsmPrinter &p, DialectOp dialectOp) {
  p << DialectOp::getOperationName() << " ";

  // Print the dialect name.
  p << dialectOp.name() << " ";

  // Print the dialect body.
  p.printRegion(dialectOp.body(), false, false);
}

//===----------------------------------------------------------------------===//
// irdl::TypeOp
//===----------------------------------------------------------------------===//

static ParseResult parseTypeOp(OpAsmParser &p, OperationState &state) {
  Builder &builder = p.getBuilder();

  // Parse the type name.
  StringRef name;
  if (failed(p.parseKeyword(&name)))
    return failure();
  state.addAttribute("name", builder.getStringAttr(name));

  return success();
}

static void print(OpAsmPrinter &p, TypeOp typeOp) {
  p << TypeOp::getOperationName() << " ";

  // Print the type name.
  p << typeOp.name();
}

//===----------------------------------------------------------------------===//
// irdl::OpTypeDefAttr
//===----------------------------------------------------------------------===//

namespace {

/// Parse a type constraint.
/// The verifier ensures that the format is respected.
ParseResult parseTypeConstraint(OpAsmParser &p, Attribute *typeConstraint) {
  Type type;

  // If the type is already registered, parse it.
  auto typeParseRes = p.parseOptionalType(type);
  if (typeParseRes.hasValue()) {
    if (typeParseRes.getValue())
      return failure();

    *typeConstraint =
        EqTypeConstraintAttr::get(*p.getBuilder().getContext(), type);
    return success();
  }

  // Otherwise, parse a dynamic type.
  StringRef name;
  if (p.parseOptionalKeyword(&name)) {
    p.emitError(p.getCurrentLocation(), "type expected");
    return failure();
  }

  *typeConstraint =
      EqDynTypeConstraintAttr::get(*p.getBuilder().getContext(), name);
  return success();
}

/// Print a type constraint.
void printTypeConstraint(OpAsmPrinter &p, Attribute typeConstraint) {
  if (auto eqConstr = typeConstraint.dyn_cast<EqTypeConstraintAttr>()) {
    p << eqConstr.getValue();
    return;
  } else if (auto eqConstr =
                 typeConstraint.dyn_cast<EqDynTypeConstraintAttr>()) {
    p << eqConstr.getValue();
    return;
  }
  assert(false && "Unknown type constraint.");
}

/// Parse an ArgDef with format "name: typeConstraint".
ParseResult parseArgDef(OpAsmParser &p, ArgDef *argDef) {
  if (p.parseKeyword(&argDef->first) || p.parseColon() ||
      parseTypeConstraint(p, &argDef->second))
    return failure();

  return success();
}

/// Print an ArgDef with format "name: typeConstraint".
void printTypedVar(OpAsmPrinter &p, const ArgDef *argDef) {
  p << argDef->first << ": ";
  printTypeConstraint(p, argDef->second);
}

/// Parse an ArgDefs with format (argDef1, argDef2, ..., argDefN).
/// The trailing comma is optional.
ParseResult parseArgDefs(OpAsmParser &p, OwningArgDefs *argDefs) {
  if (p.parseLParen())
    return failure();

  // Empty
  if (!p.parseOptionalRParen())
    return success();

  ArgDef argDef;
  if (parseArgDef(p, &argDef))
    return failure();
  argDefs->push_back(argDef);

  while (p.parseOptionalRParen()) {
    if (p.parseComma())
      return failure();

    ArgDef argDef;
    if (parseArgDef(p, &argDef))
      return failure();
    argDefs->push_back(argDef);
  }

  return success();
}

void printArgDefs(OpAsmPrinter &p, ArgDefs typedVars) {
  p << "(";
  for (size_t i = 0; i + 1 < typedVars.size(); i++) {
    const auto &typedVar = typedVars[i];
    printTypedVar(p, &typedVar);
    p << ", ";
  }
  if (typedVars.size() != 0)
    printTypedVar(p, &typedVars[typedVars.size() - 1]);
  p << ")";
}

/// Parse an OpTypeDefAttr.
/// The format is "operandDef -> resultDef" where operandDef and resultDef have
/// the ArgDefs format.
ParseResult parseOpTypeDefAttr(OpAsmParser &p, OpTypeDefAttr *opTypeDefAttr) {
  OwningArgDefs operandDefs, resultDefs;
  auto *ctx = p.getBuilder().getContext();
  // Parse the operands.
  if (parseArgDefs(p, &operandDefs))
    return failure();

  if (p.parseArrow())
    return failure();

  // Parse the results.
  if (parseArgDefs(p, &resultDefs))
    return failure();

  *opTypeDefAttr = OpTypeDefAttr::get(*ctx, operandDefs, resultDefs);

  return success();
}

void printOpTypeDef(OpAsmPrinter &p, OpTypeDef opDef) {
  printArgDefs(p, opDef.operandDef);
  p << " -> ";
  printArgDefs(p, opDef.resultDef);
}

} // namespace

//===----------------------------------------------------------------------===//
// irdl::OperationOp
//===----------------------------------------------------------------------===//

static ParseResult parseOperationOp(OpAsmParser &p, OperationState &state) {
  Builder &builder = p.getBuilder();

  // Parse the operation name.
  StringRef name;
  if (p.parseKeyword(&name))
    return failure();
  state.addAttribute("name", builder.getStringAttr(name));

  // Parse the OpDefAttr.
  OpTypeDefAttr opTypeDef;
  if (parseOpTypeDefAttr(p, &opTypeDef))
    return failure();
  state.addAttribute("op_def", opTypeDef);

  return success();
}

static void print(OpAsmPrinter &p, OperationOp operationOp) {
  p << OperationOp::getOperationName() << " ";

  // Print the operation name.
  p << operationOp.name();

  // Print the operation type constraints.
  printOpTypeDef(p, operationOp.op_def());
}

//===----------------------------------------------------------------------===//
// IRDL operations.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dyn/Dialect/IRDL/IR/IRDLOps.cpp.inc"

//===----------------------------------------------------------------------===//
// IRDL interfaces.
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDLInterface.cpp.inc"
