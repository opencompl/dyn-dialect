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
#include "mlir/IR/Builders.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

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
  registerAttributes();
}

//===----------------------------------------------------------------------===//
// Type constraints.
//===----------------------------------------------------------------------===//

namespace {
ParseResult parseTypeConstraint(OpAsmParser &p, Attribute *typeConstraint);
void printTypeConstraint(OpAsmPrinter &p, Attribute typeConstraint);

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

/// Parse an AnyOf constraint if there is one.
/// It has the format 'irdl.AnyOf<type (, type)*>'
Optional<ParseResult>
parseOptionalDynTypeParamsConstraint(OpAsmParser &p,
                                     Attribute *typeConstraint) {
  StringRef keyword;
  if (p.parseOptionalKeyword(&keyword))
    return {};

  auto loc = p.getCurrentLocation();
  auto ctx = p.getBuilder().getContext();
  auto splittedNames = keyword.split('.');
  auto dialectName = splittedNames.first;
  auto typeName = splittedNames.second;

  // Check that the type name is in the format dialectname.typename
  if (typeName == "") {
    p.emitError(loc, " expected type name prefixed with the dialect name");
    return {failure()};
  }

  // Get the dialect by its name
  auto dialect = ctx->getOrLoadDialect(dialectName);
  if (!dialect) {
    p.emitError(loc).append("dialect ", dialectName, " is not defined");
    return {failure()};
  }

  // Check that the dialect is an extensible dialect
  auto extensibleDialect = llvm::dyn_cast<ExtensibleDialect>(dialect);
  if (!extensibleDialect) {
    p.emitError(loc).append("dialect ", dialectName,
                            " is not extensible, and thus cannot be used on a "
                            "parameters constraint");
    return {failure()};
  }

  // Get the dynamic type by its name in the dialect
  auto typeDef = extensibleDialect->lookupTypeDefinition(typeName);
  if (!typeDef) {
    p.emitError(loc).append("dynamic type '", typeName,
                            "' was not registered in ", dialectName);
    return {failure()};
  }

  // Empty case, we can return an equality constraint instead of a parameters
  // constraint
  if (p.parseOptionalLess() || !p.parseOptionalGreater()) {
    *typeConstraint = EqTypeConstraintAttr::get(p.getBuilder().getContext(),
                                                DynamicType::get(typeDef));
    return {success()};
  }

  SmallVector<Attribute> paramConstraints;

  paramConstraints.push_back({});
  if (parseTypeConstraint(p, &paramConstraints.back()))
    return {failure()};

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return {failure()};

    paramConstraints.push_back({});
    if (parseTypeConstraint(p, &paramConstraints.back()))
      return {failure()};
  }

  *typeConstraint =
      DynTypeParamsConstraintAttr::get(ctx, typeDef, paramConstraints);
  return {success()};
}

void printDynTypeParamsConstraint(OpAsmPrinter &p,
                                  DynTypeParamsConstraintAttr constraint) {
  auto *typeDef = constraint.getTypeDef();
  p << typeDef->getDialect()->getNamespace() << "." << typeDef->getName();

  auto paramConstraints = constraint.getParamConstraints();
  if (paramConstraints.empty())
    return;

  p << "<";
  llvm::interleaveComma(paramConstraints, p,
                        [&p](Attribute a) { printTypeConstraint(p, a); });
  p << ">";
}

/// Parse a type constraint.
/// The verifier ensures that the format is respected.
ParseResult parseTypeConstraint(OpAsmParser &p, Attribute *typeConstraint) {
  Type type;

  // Parse an Any constraint
  auto anyRes = parseOptionalAnyTypeConstraint(p, typeConstraint);
  if (anyRes.hasValue())
    return *anyRes;

  // Parse an AnyOf constraint
  auto anyOfRes = parseOptionalAnyOfTypeConstraint(p, typeConstraint);
  if (anyOfRes.hasValue())
    return *anyOfRes;

  auto paramRes = parseOptionalDynTypeParamsConstraint(p, typeConstraint);
  if (paramRes.hasValue())
    return *paramRes;

  // Type equality constraint.
  // It has the format 'type'.
  auto typeParsed = p.parseOptionalType(type);
  if (!typeParsed.hasValue()) {
    p.emitError(p.getCurrentLocation(), "type constraint expected");
    return failure();
  }

  if (failed(typeParsed.getValue()))
    return failure();

  *typeConstraint =
      EqTypeConstraintAttr::get(p.getBuilder().getContext(), type);
  return success();
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
  } else if (auto dynTypeParamsConstr =
                 typeConstraint.dyn_cast<DynTypeParamsConstraintAttr>()) {
    printDynTypeParamsConstraint(p, dynTypeParamsConstr);
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
void printArgDef(OpAsmPrinter &p, const ArgDef *argDef) {
  p << argDef->first << ": ";
  printTypeConstraint(p, argDef->second);
}
} // namespace

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
  ctx->loadDynamicDialect(name);

  auto *dialect =
      llvm::dyn_cast<ExtensibleDialect>(ctx->getLoadedDialect(name));
  assert(dialect && "extensible dialect should have been registered");

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

/// Parse the type parameters with the format "(<argdef (, argdef)*>)?"
ParseResult parseTypeParams(OpAsmParser &p, OwningArgDefs *argDefs) {
  // No parameters
  if (p.parseOptionalLess() || !p.parseOptionalGreater())
    return success();

  ArgDef argDef;
  if (parseArgDef(p, &argDef))
    return failure();
  argDefs->push_back(argDef);

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return failure();

    ArgDef argDef;
    if (parseArgDef(p, &argDef))
      return failure();
    argDefs->push_back(argDef);
  }

  return success();
}

void printTypeParams(OpAsmPrinter &p, ArgDefs params) {
  if (params.empty()) {
    return;
  }
  p << "<";
  for (size_t i = 0; i + 1 < params.size(); i++) {
    const auto &typedVar = params[i];
    printArgDef(p, &typedVar);
    p << ", ";
  }
  printArgDef(p, &params.back());
  p << ">";
}

static ParseResult parseTypeOp(OpAsmParser &p, OperationState &state) {
  // Parse the type name.
  StringRef name;
  if (failed(p.parseKeyword(&name)))
    return failure();

  OwningArgDefs params;
  if (failed(parseTypeParams(p, &params)))
    return failure();

  auto *ctx = state.getContext();
  auto typeDef = TypeDefAttr::get(ctx, {name, params});
  state.addAttribute("def", typeDef);

  // Get the currently parsed dialect, and register the type in it.
  auto *irdlDialect = ctx->getOrLoadDialect<irdl::IRDLDialect>();
  auto *dialect = irdlDialect->currentlyParsedDialect;
  assert(dialect != nullptr && "Trying to parse an 'irdl.type' when there is "
                               "no 'irdl.dialect' currently being parsed.");
  registerType(dialect, typeDef.getTypeDef());

  return success();
}

static void print(OpAsmPrinter &p, TypeOp typeOp) {
  auto typeDef = typeOp.def().getTypeDef();
  p << TypeOp::getOperationName() << " " << typeDef.name;

  printTypeParams(p, typeDef.paramDefs);
}

//===----------------------------------------------------------------------===//
// irdl::OpDefAttr
//===----------------------------------------------------------------------===//

namespace {

/// Parse an ArgDefs with format (argDef1, argDef2, ..., argDefN).
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
    printArgDef(p, &typedVar);
    p << ", ";
  }
  if (typedVars.size() != 0)
    printArgDef(p, &typedVars[typedVars.size() - 1]);
  p << ")";
}

ParseResult parseTraitDef(OpAsmParser &p,
                          std::pair<std::string, DynamicOpTrait *> &trait) {
  auto ctx = p.getBuilder().getContext();
  auto loc = p.getCurrentLocation();

  StringRef traitName;
  if (p.parseKeyword(&traitName)) {
    p.emitError(loc, "expected trait name");
    return failure();
  }

  auto res = ctx->getDynamicTrait(traitName);
  if (!res) {
    p.emitError(loc, "trait '")
        .append(traitName, "' is not registered in the context");
    return failure();
  }
  trait.first = traitName.str();
  trait.second = res;

  return success();
}

/// Parse a TraitDefs with format '(traits [(name,)*])?'.
ParseResult parseTraitDefs(OpAsmParser &p, OwningTraitDefs &traitDefs) {
  // If the trait keyword is not present, then it means that no traits is
  // defined.
  if (p.parseOptionalKeyword("traits"))
    return success();

  if (p.parseLSquare())
    return failure();

  // Empty
  if (!p.parseOptionalRSquare())
    return success();

  std::pair<std::string, DynamicOpTrait *> trait;
  if (parseTraitDef(p, trait))
    return failure();
  traitDefs.push_back(std::move(trait));

  while (p.parseOptionalRSquare()) {
    if (p.parseComma())
      return failure();

    std::pair<std::string, DynamicOpTrait *> trait;
    if (parseTraitDef(p, trait))
      return failure();
    traitDefs.push_back(trait);
  }

  return success();
}

void printTraitDefs(OpAsmPrinter &p, TraitDefs traitDefs) {
  if (traitDefs.empty())
    return;
  p << "traits [";
  for (size_t i = 0; i + 1 < traitDefs.size(); i++) {
    p << traitDefs[i].first << ", ";
  }
  p << traitDefs.back().first << "]";
}

/// Parse an OpDefAttr.
/// The format is "operandDef -> resultDef (traits (name)+)?" where operandDef
/// and resultDef have the ArgDefs format.
ParseResult parseOpDefAttr(OpAsmParser &p, OpDefAttr *opDefAttr) {
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
  if (parseTraitDefs(p, traitDefs))
    return failure();

  *opDefAttr = OpDefAttr::get(ctx, {operandDefs, resultDefs, traitDefs});

  return success();
}

void printOpDef(OpAsmPrinter &p, OpDef opDef) {
  printArgDefs(p, opDef.operandDef);
  p << " -> ";
  printArgDefs(p, opDef.resultDef);
  if (!opDef.traitDefs.empty()) {
    p << " ";
    printTraitDefs(p, opDef.traitDefs);
  }
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
  OpDefAttr opDef;
  if (parseOpDefAttr(p, &opDef))
    return failure();
  state.addAttribute("op_def", opDef);

  // Get the currently parsed dialect
  auto *ctx = p.getBuilder().getContext();
  auto *irdlDialect = ctx->getOrLoadDialect<irdl::IRDLDialect>();
  auto *dialect = irdlDialect->currentlyParsedDialect;
  assert(dialect != nullptr &&
         "Trying to parse an 'irdl.operation' when there is "
         "no 'irdl.dialect' currently being parsed.");

  // and register the operation in the dialect
  registerOperation(dialect, name, opDef.getOpDef());

  return success();
}

static void print(OpAsmPrinter &p, OperationOp operationOp) {
  p << OperationOp::getOperationName() << " ";

  // Print the operation name.
  p << operationOp.name();

  // Print the operation type constraints.
  printOpDef(p, operationOp.op_def().getOpDef());
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
