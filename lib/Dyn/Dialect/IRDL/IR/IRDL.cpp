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
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::irdl;

//===----------------------------------------------------------------------===//
// IRDL dialect.
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDLDialect.cpp.inc"

void IRDLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dyn/Dialect/IRDL/IR/IRDLOps.cpp.inc"
      >();
  registerAttributes();
}

void IRDLDialect::addTypeWrapper(std::unique_ptr<TypeWrapper> wrapper) {
  auto emplaced =
      typeWrappers.try_emplace(wrapper->getName(), std::move(wrapper)).second;
  assert(emplaced && "a type wrapper with the same name already exists");
}

TypeWrapper *IRDLDialect::getTypeWrapper(StringRef typeName) {
  auto it = typeWrappers.find(typeName);
  if (it == typeWrappers.end())
    return nullptr;
  return it->second.get();
}

//===----------------------------------------------------------------------===//
// Type constraints.
//===----------------------------------------------------------------------===//

namespace {
ParseResult parseTypeConstraint(OpAsmParser &p, Attribute *typeConstraint,
                                ArgDefs variables);
void printTypeConstraint(OpAsmPrinter &p, Attribute typeConstraint,
                         ArgDefs variables);

/// Parse an Any constraint if there is one.
/// It has the format 'irdl.Any'
OptionalParseResult parseOptionalAnyTypeConstraint(OpAsmParser &p,
                                                   Attribute *typeConstraint) {
  if (p.parseOptionalKeyword("irdl.Any"))
    return {};

  *typeConstraint = AnyTypeConstraintAttr::get(p.getBuilder().getContext());
  return {success()};
}

/// Parse an AnyOf constraint if there is one.
/// It has the format 'irdl.AnyOf<type (, type)*>'
OptionalParseResult
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

/// Parse a type parameters constraint.
/// It has the format 'dialectname.typename<(typeConstraint ,)*>'
ParseResult parseTypeParamsConstraint(OpAsmParser &p, TypeWrapper *wrapper,
                                      Attribute *typeConstraint,
                                      ArgDefs variables) {
  auto ctx = p.getBuilder().getContext();

  // Empty case
  if (p.parseOptionalLess() || !p.parseOptionalGreater()) {
    *typeConstraint = TypeParamsConstraintAttr::get(ctx, wrapper, {});
    return success();
  }

  SmallVector<Attribute> paramConstraints;

  paramConstraints.push_back({});
  if (parseTypeConstraint(p, &paramConstraints.back(), variables))
    return {failure()};

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return {failure()};

    paramConstraints.push_back({});
    if (parseTypeConstraint(p, &paramConstraints.back(), variables))
      return {failure()};
  }

  *typeConstraint =
      TypeParamsConstraintAttr::get(ctx, wrapper, paramConstraints);
  return {success()};
}

void printTypeParamsConstraint(OpAsmPrinter &p,
                               TypeParamsConstraintAttr constraint,
                               ArgDefs variables) {
  auto *typeDef = constraint.getTypeDef();
  p << typeDef->getName();

  auto paramConstraints = constraint.getParamConstraints();
  if (paramConstraints.empty())
    return;

  p << "<";
  llvm::interleaveComma(paramConstraints, p, [&p, variables](Attribute a) {
    printTypeConstraint(p, a, variables);
  });
  p << ">";
}

/// Parse a dynamic type parameters constraint.
/// It has the format 'dialectname.typename<(typeConstraint ,)*>'
OptionalParseResult
parseOptionalDynTypeParamsConstraint(OpAsmParser &p, StringRef keyword,
                                     Attribute *typeConstraint,
                                     ArgDefs variables) {
  auto loc = p.getCurrentLocation();
  auto ctx = p.getBuilder().getContext();
  auto splittedNames = keyword.split('.');
  auto typeName = splittedNames.second;

  // Check that the type name is in the format dialectname.typename
  if (typeName == "") {
    p.emitError(loc, " expected type name prefixed with the dialect name");
    return {failure()};
  }

  if (p.parseOptionalLess() || !p.parseOptionalGreater()) {
    *typeConstraint = DynTypeParamsConstraintAttr::get(ctx, keyword, {});
    return {success()};
  }

  SmallVector<Attribute> paramConstraints;

  paramConstraints.push_back({});
  if (parseTypeConstraint(p, &paramConstraints.back(), variables))
    return {failure()};

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return {failure()};

    paramConstraints.push_back({});
    if (parseTypeConstraint(p, &paramConstraints.back(), variables))
      return {failure()};
  }

  *typeConstraint =
      DynTypeParamsConstraintAttr::get(ctx, keyword, paramConstraints);
  return {success()};
}

void printDynTypeParamsConstraint(OpAsmPrinter &p,
                                  DynTypeParamsConstraintAttr constraint,
                                  ArgDefs variables) {
  auto typeName = constraint.getTypeName();
  p << typeName;

  auto paramConstraints = constraint.getParamConstraints();
  if (paramConstraints.empty())
    return;

  p << "<";
  llvm::interleaveComma(paramConstraints, p, [&p, variables](Attribute a) {
    printTypeConstraint(p, a, variables);
  });
  p << ">";
}

/// Parse a type constraint.
/// The verifier ensures that the format is respected.
ParseResult parseTypeConstraint(OpAsmParser &p, Attribute *typeConstraint,
                                ArgDefs variables) {
  auto loc = p.getCurrentLocation();

  // Parse an Any constraint.
  auto anyRes = parseOptionalAnyTypeConstraint(p, typeConstraint);
  if (anyRes.hasValue())
    return *anyRes;

  // Parse an AnyOf constraint.
  auto anyOfRes = parseOptionalAnyOfTypeConstraint(p, typeConstraint);
  if (anyOfRes.hasValue())
    return *anyOfRes;

  auto ctx = p.getBuilder().getContext();

  // Type equality constraint.
  // It has the format 'type'.
  Type type;
  auto typeParsed = p.parseOptionalType(type);
  if (typeParsed.hasValue()) {
    if (failed(typeParsed.getValue()))
      return failure();

    *typeConstraint = EqTypeConstraintAttr::get(ctx, type);
    return success();
  }

  StringRef keyword;
  if (succeeded(p.parseOptionalKeyword(&keyword))) {
    // Check if the constraint is a type constraint variable
    for (size_t i = 0; i < variables.size(); i++)
      if (variables[i].first == keyword) {
        *typeConstraint = VarTypeConstraintAttr::get(ctx, i);
        return success();
      }

    // Parse a non-dynamic type parameter constraint.
    auto irdl = ctx->getOrLoadDialect<IRDLDialect>();
    auto typeWrapper = irdl->getTypeWrapper(keyword);
    if (typeWrapper)
      return parseTypeParamsConstraint(p, typeWrapper, typeConstraint,
                                       variables);

    // Parse a dynamic type parameter constraint.
    auto paramRes = parseOptionalDynTypeParamsConstraint(
        p, keyword, typeConstraint, variables);
    if (paramRes.hasValue())
      return *paramRes;
  }

  p.emitError(loc, "type constraint expected");
  return failure();
}

/// Print a type constraint.
void printTypeConstraint(OpAsmPrinter &p, Attribute typeConstraint,
                         ArgDefs variables) {
  if (auto eqConstr = typeConstraint.dyn_cast<EqTypeConstraintAttr>()) {
    p << eqConstr.getType();
  } else if (auto anyConstr =
                 typeConstraint.dyn_cast<AnyTypeConstraintAttr>()) {
    p << "irdl.Any";
  } else if (auto anyOfConstr =
                 typeConstraint.dyn_cast<AnyOfTypeConstraintAttr>()) {
    printAnyOfTypeConstraint(p, anyOfConstr);
  } else if (auto typeParamsConstr =
                 typeConstraint.dyn_cast<TypeParamsConstraintAttr>()) {
    printTypeParamsConstraint(p, typeParamsConstr, variables);
  } else if (auto dynTypeParamsConstr =
                 typeConstraint.dyn_cast<DynTypeParamsConstraintAttr>()) {
    printDynTypeParamsConstraint(p, dynTypeParamsConstr, variables);
  } else if (auto typeConstraintParam =
                 typeConstraint.dyn_cast<VarTypeConstraintAttr>()) {
    p << variables[typeConstraintParam.getIndex()].first;
  } else {
    assert(false && "Unknown type constraint.");
  }
}

/// Parse an ArgDef with format "name: typeConstraint".
ParseResult parseArgDef(OpAsmParser &p, ArgDef *argDef, ArgDefs variables) {
  if (p.parseKeyword(&argDef->first) || p.parseColon() ||
      parseTypeConstraint(p, &argDef->second, variables))
    return failure();

  return success();
}

/// Print an ArgDef with format "name: typeConstraint".
void printArgDef(OpAsmPrinter &p, ArgDef argDef, ArgDefs variables) {
  p << argDef.first << ": ";
  printTypeConstraint(p, argDef.second, variables);
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

  // Parse the dialect body.
  Region *region = state.addRegion();
  if (failed(p.parseRegion(*region)))
    return failure();

  DialectOp::ensureTerminator(*region, builder, state.location);

  return success();
}

static void print(OpAsmPrinter &p, DialectOp dialectOp) {
  // Print the dialect name.
  p << " " << dialectOp.name() << " ";

  // Print the dialect body.
  p.printRegion(dialectOp.body(), false, false);
}

//===----------------------------------------------------------------------===//
// NamedTypeConstraintArray
//===----------------------------------------------------------------------===//

ParseResult parseNamedTypeConstraint(OpAsmParser &p,
                                     NamedTypeConstraintAttr &param) {
  std::string name;
  if (failed(p.parseKeywordOrString(&name)))
    return failure();
  if (failed(p.parseColon()))
    return failure();
  Attribute attr;
  if (failed(parseTypeConstraint(p, &attr, {})))
    return failure();
  param = NamedTypeConstraintAttr::get(p.getContext(), name, attr);
  return success();
}

void printNamedTypeConstraint(OpAsmPrinter &p, NamedTypeConstraintAttr attr) {
  p.printKeywordOrString(attr.getName());
  p << ": ";
  printTypeConstraint(p, attr.getConstraint(), {});
}

ParseResult parseNamedTypeConstraintArray(OpAsmParser &p,
                                          ArrayAttr &paramsAttr) {
  SmallVector<Attribute> attrs;
  auto parseRes = p.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        NamedTypeConstraintAttr attr;
        if (failed(parseNamedTypeConstraint(p, attr)))
          return failure();
        attrs.push_back(attr);
        return success();
      });
  if (parseRes.failed())
    return failure();
  paramsAttr = ArrayAttr::get(p.getContext(), attrs);
  return success();
}

void printNamedTypeConstraintArray(OpAsmPrinter &p, Operation *,
                                   ArrayAttr paramsAttr) {
  p << "(";
  llvm::interleaveComma(paramsAttr.getValue(), p, [&](Attribute attr) {
    printNamedTypeConstraint(p, attr.cast<NamedTypeConstraintAttr>());
  });
  p << ")";
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
  if (parseArgDef(p, &argDef, {}))
    return failure();
  argDefs->push_back(argDef);

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return failure();

    ArgDef argDef;
    if (parseArgDef(p, &argDef, {}))
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
    printArgDef(p, typedVar, {});
    p << ", ";
  }
  printArgDef(p, params.back(), {});
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

  // Parse the type definition region.
  auto *region = state.addRegion();
  auto regionParseRes = p.parseOptionalRegion(*region);
  if (regionParseRes.hasValue()) {
    if (failed(regionParseRes.getValue()))
      return failure();
  }
  // If no regions are parsed, add a single empty block to the operation
  // region.
  if (region->getBlocks().size() == 0) {
    region->push_back(new Block());
  }

  auto *ctx = state.getContext();
  auto typeDef = TypeDefAttr::get(ctx, {name, params});
  state.addAttribute("def", typeDef);

  return success();
}

static void print(OpAsmPrinter &p, TypeOp typeOp) {
  auto typeDef = typeOp.def().getTypeDef();
  p << " " << typeDef.name << " ";

  printTypeParams(p, typeDef.paramDefs);

  if (!typeOp.body().getBlocks().front().empty()) {
    p.printRegion(typeOp.body());
  }
}

//===----------------------------------------------------------------------===//
// irdl::OpDefAttr
//===----------------------------------------------------------------------===//

namespace {

ParseResult parseTypeConstraintVars(OpAsmParser &p,
                                    OwningArgDefs *typeConstrVars) {
  // No parameters
  if (p.parseOptionalLess() || !p.parseOptionalGreater())
    return success();

  ArgDef argDef;
  if (parseArgDef(p, &argDef, *typeConstrVars))
    return failure();
  typeConstrVars->push_back(argDef);

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return failure();

    ArgDef argDef;
    if (parseArgDef(p, &argDef, *typeConstrVars))
      return failure();
    typeConstrVars->push_back(argDef);
  }

  return success();
}

void printTypeConstraintVars(OpAsmPrinter &p, ArgDefs typeConstrVars) {
  if (typeConstrVars.empty()) {
    return;
  }

  p << "<";
  for (size_t i = 0; i + 1 < typeConstrVars.size(); i++) {
    const auto &typedVar = typeConstrVars[i];
    printArgDef(p, typedVar, typeConstrVars);
    p << ", ";
  }
  printArgDef(p, typeConstrVars.back(), typeConstrVars);
  p << ">";
}

/// Parse an ArgDefs with format (argDef1, argDef2, ..., argDefN).
ParseResult parseArgDefs(OpAsmParser &p, OwningArgDefs *argDefs,
                         ArgDefs typeConstrVars) {
  if (p.parseLParen())
    return failure();

  // Empty
  if (!p.parseOptionalRParen())
    return success();

  ArgDef argDef;
  if (parseArgDef(p, &argDef, typeConstrVars))
    return failure();
  argDefs->push_back(argDef);

  while (p.parseOptionalRParen()) {
    if (p.parseComma())
      return failure();

    ArgDef argDef;
    if (parseArgDef(p, &argDef, typeConstrVars))
      return failure();
    argDefs->push_back(argDef);
  }

  return success();
}

void printArgDefs(OpAsmPrinter &p, ArgDefs typedVars, ArgDefs variables) {
  p << "(";
  for (size_t i = 0; i + 1 < typedVars.size(); i++) {
    const auto &typedVar = typedVars[i];
    printArgDef(p, typedVar, variables);
    p << ", ";
  }
  if (typedVars.size() != 0)
    printArgDef(p, typedVars[typedVars.size() - 1], variables);
  p << ")";
}

/// Parse an OpDefAttr.
/// The format is "operandDef -> resultDef (traits (name)+)?" where operandDef
/// and resultDef have the ArgDefs format.
ParseResult parseOpDefAttr(OpAsmParser &p, OpDefAttr *opDefAttr) {
  OwningArgDefs typeConstrVars, operandDefs, resultDefs;
  auto *ctx = p.getBuilder().getContext();
  // Parse the type constraint variables
  if (parseTypeConstraintVars(p, &typeConstrVars))
    return failure();

  // Parse the operands.
  if (parseArgDefs(p, &operandDefs, typeConstrVars))
    return failure();

  if (p.parseArrow())
    return failure();

  // Parse the results.
  if (parseArgDefs(p, &resultDefs, typeConstrVars))
    return failure();

  *opDefAttr = OpDefAttr::get(ctx, {typeConstrVars, operandDefs, resultDefs});

  return success();
}

void printOpDef(OpAsmPrinter &p, OpDef opDef) {
  printTypeConstraintVars(p, opDef.typeConstraintVars);
  printArgDefs(p, opDef.operandDef, opDef.typeConstraintVars);
  p << " -> ";
  printArgDefs(p, opDef.resultDef, opDef.typeConstraintVars);
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

  return success();
}

static void print(OpAsmPrinter &p, OperationOp operationOp) {
  p << " ";

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
