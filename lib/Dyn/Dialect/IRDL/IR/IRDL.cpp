//===- IRDL.cpp - IRDL dialect ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
#include "Dyn/Dialect/IRDL/IR/StandardOpInterfaces.h"
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
  registerAttributes();
  registerStandardInterfaceAttributes();
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
  auto *ctx = state.getContext();
  auto *dynCtx = ctx->getOrLoadDialect<DynamicContext>();
  auto dialectRes = dynCtx->createAndRegisterDialect(name);
  if (failed(dialectRes))
    return failure();
  auto *dialect = *dialectRes;

  // Set the current dialect to the dialect that we are currently defining.
  // Every IRDL operation that is parsed in the next region will be registered
  // inside this dialect.
  auto irdlDialect = ctx->getLoadedDialect<irdl::IRDLDialect>();
  irdlDialect->currentlyParsedDialect = dialect;

  // Parse the dialect body.
  Region *region = state.addRegion();
  if (failed(p.parseRegion(*region)))
    return failure();

  // We are not parsing the dialect anymore.
  irdlDialect->currentlyParsedDialect = nullptr;

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
  auto *ctx = state.getContext();
  auto *irdlDialect = ctx->getOrLoadDialect<irdl::IRDLDialect>();
  auto *dialect = irdlDialect->currentlyParsedDialect;
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

/// Parse an Any constraint if there is one.
/// It has the format 'irdl.Any'
Optional<ParseResult>
parseOptionalAnyTypeConstraint(OpAsmParser &p, Attribute *typeConstraint) {
  if (p.parseOptionalKeyword("irdl.Any"))
    return {};

  *typeConstraint = AnyTypeConstraintAttr::get(p.getBuilder().getContext());
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

  if (p.parseType(type))
    return {failure()};
  types.push_back(type);

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return {failure()};

    Type type;
    if (p.parseType(type))
      return {failure()};
    types.push_back(type);
  }

  *typeConstraint =
      AnyOfTypeConstraintAttr::get(p.getBuilder().getContext(), types);
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
  auto typeParsed = p.parseOptionalType(type);
  if (!typeParsed.hasValue()) {
    p.emitError(p.getCurrentLocation(), "type constraint expected");
  }

  if (failed(typeParsed.getValue()))
    return failure();

  *typeConstraint =
      EqTypeConstraintAttr::get(p.getBuilder().getContext(), type);
  return success();
}

/// Print an AnyOf type constraint.
/// It has the format 'irdl.AnyOf<type, (, type)*>'.
void printAnyOfTypeConstraint(OpAsmPrinter &p,
                              AnyOfTypeConstraintAttr anyOfConstr) {
  auto types = anyOfConstr.getTypes();

  p << "irdl.AnyOf<";
  for (size_t i = 0; i + 1 < types.size(); i++) {
    p << types[i] << ", ";
  }
  p << types.back() << ">";
}

/// Print a type constraint.
void printTypeConstraint(OpAsmPrinter &p, Attribute typeConstraint) {
  if (auto eqConstr = typeConstraint.dyn_cast<EqTypeConstraintAttr>()) {
    p << eqConstr.getType();
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

ParseResult parseTraitDef(OpAsmParser &p, DynamicContext *dynCtx,
                          DynamicOpTrait **trait) {
  auto loc = p.getCurrentLocation();

  StringRef traitName;
  if (p.parseKeyword(&traitName)) {
    p.emitError(loc, "expected trait name");
    return failure();
  }

  auto res = dynCtx->lookupOpTrait(traitName);
  if (failed(res)) {
    p.emitError(loc, "trait '").append(traitName, "' is not defined");
    return failure();
  }
  *trait = *res;

  return success();
}

/// Parse an interface implementation.
/// This will first parse an interface name, then get the interface definition,
/// and use its custom parser to parse the implementation.
ParseResult parseInterfaceDef(OpAsmParser &p, DynamicContext *dynCtx,
                              InterfaceImplAttrInterface *interface) {
  auto loc = p.getCurrentLocation();

  StringRef interfaceName;
  if (p.parseKeyword(&interfaceName)) {
    p.emitError(loc, "expected interface name");
    return failure();
  }

  auto interfaceRes = dynCtx->lookupOpInterface(interfaceName);
  if (failed(interfaceRes)) {
    p.emitError(loc, "interface '")
        .append(interfaceName, "' is not registered");
    return failure();
  }

  return (*interfaceRes)->parseImpl(p, *interface);
}

/// Parse a TraitDefs with format '(traits [(name,)*])?'.
ParseResult parseTraitDefs(OpAsmParser &p, OwningTraitDefs *traitDefs) {
  // If the trait keyword is not present, then it means that no traits is
  // defined.
  if (p.parseOptionalKeyword("traits"))
    return success();

  if (p.parseLSquare())
    return failure();

  // Empty
  if (!p.parseOptionalRSquare())
    return success();

  auto *dynCtx =
      p.getBuilder().getContext()->getLoadedDialect<DynamicContext>();

  DynamicOpTrait *trait;
  if (parseTraitDef(p, dynCtx, &trait))
    return failure();
  traitDefs->push_back(trait);

  while (p.parseOptionalRSquare()) {
    if (p.parseComma())
      return failure();

    DynamicOpTrait *trait;
    if (parseTraitDef(p, dynCtx, &trait))
      return failure();
    traitDefs->push_back(trait);
  }

  return success();
}

void printTraitDefs(OpAsmPrinter &p, TraitDefs traitDefs) {
  if (traitDefs.empty())
    return;
  p << "traits [";
  for (size_t i = 0; i + 1 < traitDefs.size(); i++) {
    p << traitDefs[i]->name;
    p << ", ";
  }
  p << traitDefs.back()->name << "]";
}

/// Parse an InterfaceDefs with format '(interfaces [(interface,)*])?'.
ParseResult parseInterfaceDefs(OpAsmParser &p,
                               OwningInterfaceDefs *interfaceDefs) {
  // If the interface keyword is not present, then it means that no interface is
  // defined.
  if (p.parseOptionalKeyword("interfaces"))
    return success();

  if (p.parseLSquare())
    return failure();

  // Empty
  if (!p.parseOptionalRSquare())
    return success();

  auto *dynCtx =
      p.getBuilder().getContext()->getLoadedDialect<DynamicContext>();

  InterfaceImplAttrInterface interface;
  if (parseInterfaceDef(p, dynCtx, &interface))
    return failure();
  interfaceDefs->push_back(interface);

  while (p.parseOptionalRSquare()) {
    if (p.parseComma())
      return failure();

    InterfaceImplAttrInterface interface;
    if (parseInterfaceDef(p, dynCtx, &interface))
      return failure();
    interfaceDefs->push_back(interface);
  }

  return success();
}

void printInterfaceDefs(OpAsmPrinter &p, InterfaceDefs interfaceDefs) {
  if (interfaceDefs.empty())
    return;
  p << "interfaces [";
  for (size_t i = 0; i + 1 < interfaceDefs.size(); i++) {
    interfaceDefs[i].print(p);
    p << ", ";
  }
  interfaceDefs.back().print(p);
  p << "]";
}

/// Parse an OpTypeDefAttr.
/// The format is "operandDef -> resultDef (traits (name)+)?" where operandDef
/// and resultDef have the ArgDefs format.
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

  // Parse the associated traits.
  OwningTraitDefs traitDefs;
  if (parseTraitDefs(p, &traitDefs))
    return failure();

  OwningInterfaceDefs interfaceDefs;
  if (parseInterfaceDefs(p, &interfaceDefs))
    return failure();

  *opTypeDefAttr = OpTypeDefAttr::get(
      ctx, {operandDefs, resultDefs, traitDefs, interfaceDefs});

  return success();
}

void printOpTypeDef(OpAsmPrinter &p, OpTypeDef opDef) {
  printArgDefs(p, opDef.operandDef);
  p << " -> ";
  printArgDefs(p, opDef.resultDef);
  if (!opDef.traitDefs.empty()) {
    p << " ";
    printTraitDefs(p, opDef.traitDefs);
  }
  if (!opDef.interfaceDefs.empty()) {
    p << " ";
    printInterfaceDefs(p, opDef.interfaceDefs);
  }
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
  if (p.parseKeyword(&name) || p.parseEqual() || p.parseType(type))
    return failure();

  state.addAttribute("name", builder.getStringAttr(name));
  state.addAttribute("type", TypeAttr::get(type));

  // Get the currently parsed dialect.

  auto *ctx = p.getBuilder().getContext();
  auto *irdlDialect = ctx->getOrLoadDialect<irdl::IRDLDialect>();
  auto *dialect = irdlDialect->currentlyParsedDialect;
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
  auto *ctx = p.getBuilder().getContext();
  auto *irdlDialect = ctx->getOrLoadDialect<irdl::IRDLDialect>();
  auto *dialect = irdlDialect->currentlyParsedDialect;
  assert(dialect != nullptr &&
         "Trying to parse an 'irdl.operation' when there is "
         "no 'irdl.dialect' currently being parsed.");

  // and register the operation in the dialect
  if (failed(registerOperation(dialect, name, opTypeDef.getOpDef())))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, OperationOp operationOp) {
  p << OperationOp::getOperationName() << " ";

  // Print the operation name.
  p << operationOp.name();

  // Print the operation type constraints.
  printOpTypeDef(p, operationOp.op_def().getOpDef());
}

//===----------------------------------------------------------------------===//
// IRDL operations.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dyn/Dialect/IRDL/IR/IRDLOps.cpp.inc"

//===----------------------------------------------------------------------===//
// IRDL interfaces.
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDLInterfaces.cpp.inc"
