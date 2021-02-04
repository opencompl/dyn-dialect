//===- IRDL.cpp - IRDL dialect ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
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
  addAttributes<OpTypeDefAttr>();
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

/// Parse a type constraint with the format "dialect.name".
/// The verifier ensures that the format is respected. Only a keyword is parsed
/// (since mlir will parse "." in keywords).
ParseResult parseTypeConstraint(OpAsmParser &p,
                                TypeConstraint *typeConstraint) {
  StringRef name;

  if (p.parseKeyword(&name))
    return failure();

  *typeConstraint = name.str();
  return success();
}

/// Print a type constraint with the format "dialect.name".
void printTypeConstraint(OpAsmPrinter &p,
                         const TypeConstraint &typeConstraint) {
  p << typeConstraint;
}

/// Parse an ArgDef with format "name: typeConstraint".
ParseResult parseArgDef(OpAsmParser &p, ArgDef *argDef) {
  StringRef name;
  TypeConstraint typeConstraint;

  if (p.parseKeyword(&name) || p.parseColon() ||
      parseTypeConstraint(p, &argDef->second))
    return failure();

  argDef->first = name.str();
  return success();
}

/// Print an ArgDef with format "name: typeConstraint".
void printTypedVar(OpAsmPrinter &p, const ArgDef &argDef) {
  p << argDef.first << ": ";
  printTypeConstraint(p, argDef.second);
}

/// Parse an ArgDefs with format ([argDef,]*).
/// The trailing comma is optional.
ParseResult parseArgDefs(OpAsmParser &p, OwningArgDefs *argDefs) {
  if (p.parseLParen())
    return failure();

  while (true) {
    if (!p.parseOptionalRParen())
      break;

    ArgDef argDef;
    if (parseArgDef(p, &argDef))
      return failure();
    argDefs->push_back(argDef);

    if (!p.parseOptionalComma())
      continue;

    if (p.parseRParen())
      return failure();

    break;
  }

  return success();
}

void printArgDefs(OpAsmPrinter &p, ArgDefs typedVars) {
  p << "(";
  for (const auto &typedVar : typedVars) {
    printTypedVar(p, typedVar);
    p << ", ";
  }
  p << ")";
}

/// Parse an OpTypeDefAttr.
/// The format is "operandDef -> resultDef" where operandDef and resultDef have
/// the ArgDefs format.
ParseResult parseOpTypeDefAttr(OpAsmParser &p, OpTypeDefAttr *opTypeDefAttr) {
  OwningArgDefs operandDefs, resultDefs;
  // Parse the operands.
  if (parseArgDefs(p, &operandDefs))
    return failure();

  if (p.parseArrow())
    return failure();

  // Parse the results.
  if (parseArgDefs(p, &resultDefs))
    return failure();

  *opTypeDefAttr =
      OpTypeDefAttr::get(*p.getBuilder().getContext(), std::move(operandDefs),
                         std::move(resultDefs));
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
