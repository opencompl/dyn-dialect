//===- IRDL.cpp - IRDL dialect ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
#include "Dyn/Dialect/IRDL/IRDLRegistration.h"
#include "Dyn/Dialect/IRDL/TypeConstraint.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::irdl;
using namespace mlir::dyn;

//===----------------------------------------------------------------------===//
// IRDL dialect.
//===----------------------------------------------------------------------===//

void IRDLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dyn/Dialect/IRDL/IR/IRDLOps.cpp.inc"
      >();
  addAttributes<OpTypeDefAttr, EqTypeConstraintAttr, AnyOfTypeConstraintAttr,
                AnyTypeConstraintAttr>();
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

  // Register the dialect in the dynamic context.
  auto *dynCtx = state.getContext()->getOrLoadDialect<DynamicContext>();
  auto dialectRes = dynCtx->createAndRegisterDialect(name);
  if (failed(dialectRes))
    return failure();
  auto *dialect = *dialectRes;

  // Set the current dialect to the dialect that we are currently defining.
  // Every IRDL operation that is parsed in the next region will be registered
  // inside this dialect.
  dynCtx->currentlyParsedDialect = dialect;

  // Parse the dialect body.
  Region *region = state.addRegion();
  if (failed(p.parseRegion(*region)))
    return failure();

  // We are not parsing the dialect anymore.
  dynCtx->currentlyParsedDialect = nullptr;

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

  // Get the currently parsed dialect, and register the type in it.
  auto *dynCtx = state.getContext()->getOrLoadDialect<DynamicContext>();
  auto *dialect = dynCtx->currentlyParsedDialect;
  assert(dialect != nullptr && "Trying to parse an 'irdl.type' when there is "
                               "no 'irdl.dialect' currently being parsed.");
  if (failed(registerType(dialect, name)))
    return failure();

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

/// Parse a optionally a type.
/// A type has either the usual MLIR format, or is a name. If the type has a
/// name (let's say `type`), it will correspond  to the type `!dialect.type`,
/// where `dialect` is the name of the previously defined dialect using
/// `irdl.dialect`.
/// Returns a ParseResult if something was parsed, and no values otherwise.
Optional<ParseResult> parseOptionalType(OpAsmParser &p, Type *type) {
  // If we can parse the type directly, do it.
  auto typeParseRes = p.parseOptionalType(*type);
  if (typeParseRes.hasValue())
    return typeParseRes.getValue();

  auto loc = p.getCurrentLocation();
  // Otherwise, this mean that the type is in the format `type` instead of
  // `dialect.type`.
  StringRef typeName;
  if (p.parseOptionalKeyword(&typeName))
    return {};

  auto dynCtx = p.getBuilder().getContext()->getLoadedDialect<DynamicContext>();
  auto *dialect = dynCtx->currentlyParsedDialect;
  assert(dialect && "Trying to parse a possible dynamic type when there is "
                    "no 'irdl.dialect' currently being parsed.");

  /// Get the type from the dialect.
  auto dynType = dialect->lookupTypeOrTypeAlias(typeName);
  if (failed(dynType))
    return ParseResult(p.emitError(loc, "type ")
                           .append(typeName,
                                   " is not registered in the dialect ",
                                   dialect->getName(), "."));

  *type = *dynType;
  return ParseResult(success());
}

/// Parse a type, and returns an error if there is none.
ParseResult parseType(OpAsmParser &p, Type *type) {
  auto res = parseOptionalType(p, type);

  if (res.hasValue())
    return res.getValue();

  p.emitError(p.getCurrentLocation(), "type expected");
  return failure();
}

/// Parse an Any constraint if there is one.
/// It has the format 'irdl.Any'
Optional<ParseResult>
parseOptionalAnyTypeConstraint(OpAsmParser &p, Attribute *typeConstraint) {
  if (p.parseOptionalKeyword("irdl.Any"))
    return {};

  *typeConstraint = AnyTypeConstraintAttr::get(*p.getBuilder().getContext());
  return {success()};
}

/// Parse an AnyOf constraint if there is one.
/// It has the format 'irdl.AnyOf<type (, type)*>'
Optional<ParseResult>
parseOptionalAnyOfTypeConstraint(OpAsmParser &p, Attribute *typeConstraint) {
  if (p.parseOptionalKeyword("irdl.AnyOf"))
    return {};

  if (p.parseLess())
    return {failure()};

  std::vector<Type> types;
  Type type;

  if (parseType(p, &type))
    return {failure()};
  types.push_back(type);

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return {failure()};

    Type type;
    if (parseType(p, &type))
      return {failure()};
    types.push_back(type);
  }

  *typeConstraint =
      AnyOfTypeConstraintAttr::get(*p.getBuilder().getContext(), types);
  return {success()};
}

/// Parse a type constraint.
/// The verifier ensures that the format is respected.
ParseResult parseTypeConstraint(OpAsmParser &p, Attribute *typeConstraint) {
  Type type;

  // Parse an Any constraint
  auto anyRes = parseOptionalAnyTypeConstraint(p, typeConstraint);
  if (anyRes.hasValue())
    return anyRes.getValue();

  // Parse an AnyOf constraint
  auto anyOfRes = parseOptionalAnyOfTypeConstraint(p, typeConstraint);
  if (anyOfRes.hasValue())
    return anyOfRes.getValue();

  // Type equality constraint.
  // It has the format 'type'.
  auto typeParsed = parseOptionalType(p, &type);
  if (!typeParsed.hasValue()) {
    p.emitError(p.getCurrentLocation(), "type constraint expected");
  }

  if (failed(typeParsed.getValue()))
    return failure();

  *typeConstraint =
      EqTypeConstraintAttr::get(*p.getBuilder().getContext(), type);
  return success();
}

/// Print an AnyOf type constraint.
/// It has the format 'irdl.AnyOf<type, (, type)*>'.
void printAnyOfTypeConstraint(OpAsmPrinter &p,
                              AnyOfTypeConstraintAttr anyOfConstr) {
  auto types = anyOfConstr.getValue();

  p << "irdl.AnyOf<";
  for (size_t i = 0; i + 1 < types.size(); i++) {
    p << types[i] << ", ";
  }
  p << types.back() << ">";
}

/// Print a type constraint.
void printTypeConstraint(OpAsmPrinter &p, Attribute typeConstraint) {
  if (auto eqConstr = typeConstraint.dyn_cast<EqTypeConstraintAttr>()) {
    p << eqConstr.getValue();
  } else if (auto anyConstr =
                 typeConstraint.dyn_cast<AnyTypeConstraintAttr>()) {
    p << "irdl.Any";
  } else if (auto anyOfConstr =
                 typeConstraint.dyn_cast<AnyOfTypeConstraintAttr>()) {
    printAnyOfTypeConstraint(p, anyOfConstr);
  } else {
    assert(false && "Unknown type constraint.");
  }
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
// irdl::TypeAliasOp
//===----------------------------------------------------------------------===//

static ParseResult parseTypeAliasOp(OpAsmParser &p, OperationState &state) {
  Builder &builder = p.getBuilder();

  // Parse the operation name.
  StringRef name;
  Type type;
  if (p.parseKeyword(&name) || p.parseEqual() || parseType(p, &type))
    return failure();

  state.addAttribute("name", builder.getStringAttr(name));
  state.addAttribute("type", TypeAttr::get(type));

  // Get the currently parsed dialect.
  auto *dynCtx = state.getContext()->getOrLoadDialect<DynamicContext>();
  auto *dialect = dynCtx->currentlyParsedDialect;
  assert(dialect != nullptr);

  // and register the type aliast in the dialect.
  return registerTypeAlias(dialect, name, type);
}

static void print(OpAsmPrinter &p, TypeAliasOp typeAliasOp) {
  p << TypeAliasOp::getOperationName() << " " << typeAliasOp.name() << " = "
    << typeAliasOp.type();
}

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

  // Get the currently parsed dialect
  auto *dynCtx = state.getContext()->getOrLoadDialect<DynamicContext>();
  auto *dialect = dynCtx->currentlyParsedDialect;
  assert(dialect != nullptr &&
         "Trying to parse an 'irdl.operation' when there is "
         "no 'irdl.dialect' currently being parsed.");

  // and register the operation in the dialect
  if (failed(registerOperation(dialect, name, opTypeDef.getValue())))
    return failure();

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
