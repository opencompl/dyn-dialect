//===- IRDL.cpp - IRDL dialect ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
#include "Dyn/Dialect/IRDL/TypeWrapper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::irdl;

using ArgDef = std::pair<StringRef, Attribute>;
using ArgDefs = ArrayRef<ArgDef>;

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
  this->irdlContext.addTypeWrapper(std::move(wrapper));
}

TypeWrapper *IRDLDialect::getTypeWrapper(StringRef typeName) {
  return this->irdlContext.getTypeWrapper(typeName);
}

//===----------------------------------------------------------------------===//
// Parsing/Printing
//===----------------------------------------------------------------------===//

static ParseResult parseKeywordOrString(OpAsmParser &p, StringAttr &attr) {
  std::string str;
  if (failed(p.parseKeywordOrString(&str)))
    return failure();
  attr = p.getBuilder().getStringAttr(str);
  return success();
}

static void printKeywordOrString(OpAsmPrinter &p, Operation *,
                                 StringAttr attr) {
  p.printKeywordOrString(attr.getValue());
}

/// Parse a region, and add a single block if the region is empty.
/// If no region is parsed, create a new region with a single empty block.
static ParseResult parseSingleBlockRegion(OpAsmParser &p, Region &region) {
  auto regionParseRes = p.parseOptionalRegion(region);
  if (regionParseRes.has_value()) {
    if (failed(regionParseRes.value()))
      return failure();
  }
  // If the region is empty, add a single empty block.
  if (region.getBlocks().size() == 0) {
    region.push_back(new Block());
  }

  return success();
}

static void printSingleBlockRegion(OpAsmPrinter &p, Operation *op,
                                   Region &region) {
  if (!region.getBlocks().front().empty()) {
    p.printRegion(region);
  }
}

//===----------------------------------------------------------------------===//
// Type constraints.
//===----------------------------------------------------------------------===//

namespace {
ParseResult parseTypeConstraint(OpAsmParser &p, Attribute *typeConstraint);
void printTypeConstraint(OpAsmPrinter &p, Attribute typeConstraint);

/// Parse an Any constraint if there is one.
/// It has the format 'Any'
OptionalParseResult parseOptionalAnyTypeConstraint(OpAsmParser &p,
                                                   Attribute *typeConstraint) {
  if (p.parseOptionalKeyword("Any"))
    return {};

  *typeConstraint = AnyTypeConstraintAttr::get(p.getBuilder().getContext());
  return {success()};
}

/// Parse an AnyOf constraint if there is one.
/// It has the format 'AnyOf<type (, type)*>'
OptionalParseResult
parseOptionalAnyOfTypeConstraint(OpAsmParser &p, Attribute *typeConstraint) {
  if (p.parseOptionalKeyword("AnyOf"))
    return {};

  if (p.parseLess())
    return {failure()};

  SmallVector<Attribute> constraints;

  {
    Attribute constraint;
    if (parseTypeConstraint(p, &constraint))
      return {failure()};
    constraints.push_back(constraint);
  }

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return {failure()};

    Attribute constraint;
    if (parseTypeConstraint(p, &constraint))
      return {failure()};
    constraints.push_back(constraint);
  }

  *typeConstraint =
      AnyOfTypeConstraintAttr::get(p.getBuilder().getContext(), constraints);
  return {success()};
}

/// Print an AnyOf type constraint.
/// It has the format 'AnyOf<type, (, type)*>'.
void printAnyOfTypeConstraint(OpAsmPrinter &p,
                              AnyOfTypeConstraintAttr anyOfConstr) {
  auto constrs = anyOfConstr.getConstrs();

  p << "AnyOf<";
  for (size_t i = 0; i + 1 < constrs.size(); i++) {
    printTypeConstraint(p, constrs[i]);
    p << ", ";
  }
  printTypeConstraint(p, constrs.back());
  p << ">";
}

/// Parse an And constraint if there is one.
/// It has the format 'And<type (, type)*>'.
OptionalParseResult parseOptionalAndTypeConstraint(OpAsmParser &p,
                                                   Attribute *typeConstraint) {
  if (p.parseOptionalKeyword("And"))
    return {};

  if (p.parseLess())
    return {failure()};

  SmallVector<Attribute> constraints;

  {
    Attribute constraint;
    if (parseTypeConstraint(p, &constraint))
      return {failure()};
    constraints.push_back(constraint);
  }

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return {failure()};

    Attribute constraint;
    if (parseTypeConstraint(p, &constraint))
      return {failure()};
    constraints.push_back(constraint);
  }

  *typeConstraint =
      AndTypeConstraintAttr::get(p.getBuilder().getContext(), constraints);
  return {success()};
}

/// Print an And type constraint.
/// It has the format 'And<type (, type)*>'.
void printAndTypeConstraint(OpAsmPrinter &p, AndTypeConstraintAttr andConstr) {
  auto constrs = andConstr.getConstrs();

  p << "And<";
  for (size_t i = 0; i + 1 < constrs.size(); i++) {
    printTypeConstraint(p, constrs[i]);
    p << ", ";
  }
  printTypeConstraint(p, constrs.back());
  p << ">";
}

/// Parse a type parameters constraint.
/// It has the format 'dialectname.typename<(typeConstraint ,)*>'
ParseResult parseTypeParamsConstraint(OpAsmParser &p, TypeWrapper *wrapper,
                                      Attribute *typeConstraint) {
  auto ctx = p.getBuilder().getContext();

  // Empty case
  if (p.parseOptionalGreater().succeeded()) {
    *typeConstraint = TypeParamsConstraintAttr::get(ctx, wrapper, {});
    return success();
  }

  SmallVector<Attribute> paramConstraints;

  paramConstraints.push_back({});
  if (parseTypeConstraint(p, &paramConstraints.back()))
    return failure();

  while (p.parseOptionalGreater()) {
    if (p.parseComma())
      return failure();

    paramConstraints.push_back({});
    if (parseTypeConstraint(p, &paramConstraints.back()))
      return failure();
  }

  *typeConstraint =
      TypeParamsConstraintAttr::get(ctx, wrapper, paramConstraints);
  return success();
}

void printTypeParamsConstraint(OpAsmPrinter &p,
                               TypeParamsConstraintAttr constraint) {
  auto *typeDef = constraint.getTypeDef();
  p << typeDef->getName();

  auto paramConstraints = constraint.getParamConstraints();

  p << "<";
  llvm::interleaveComma(paramConstraints, p,
                        [&p](Attribute a) { printTypeConstraint(p, a); });
  p << ">";
}

/// Parse a dynamic type base type constraint.
/// It has the format 'dialectname.typename'
OptionalParseResult
parseOptionalDynTypeBaseConstraint(OpAsmParser &p, StringRef keyword,
                                   Attribute *typeConstraint) {
  auto loc = p.getCurrentLocation();
  auto ctx = p.getBuilder().getContext();
  auto splittedNames = keyword.split('.');
  auto typeName = splittedNames.second;

  // Check that the type name is in the format dialectname.typename
  if (typeName == "") {
    p.emitError(loc, " expected type name prefixed with the dialect name");
    return {failure()};
  }

  *typeConstraint = DynTypeBaseConstraintAttr::get(ctx, keyword);
  return {success()};
}

void printDynTypeBaseConstraint(OpAsmPrinter &p,
                                DynTypeBaseConstraintAttr constraint) {
  auto typeName = constraint.getTypeName();
  p << typeName;
}

/// Parse a dynamic type parameters constraint.
/// It has the format 'dialectname.typename<(typeConstraint ,)*>'
OptionalParseResult
parseOptionalDynTypeParamsConstraint(OpAsmParser &p, StringRef keyword,
                                     Attribute *typeConstraint) {
  auto loc = p.getCurrentLocation();
  auto ctx = p.getBuilder().getContext();
  auto splittedNames = keyword.split('.');
  auto typeName = splittedNames.second;

  // Check that the type name is in the format dialectname.typename
  if (typeName == "") {
    p.emitError(loc, " expected type name prefixed with the dialect name");
    return {failure()};
  }

  if (!p.parseOptionalGreater()) {
    *typeConstraint = DynTypeParamsConstraintAttr::get(ctx, keyword, {});
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
      DynTypeParamsConstraintAttr::get(ctx, keyword, paramConstraints);
  return {success()};
}

void printDynTypeParamsConstraint(OpAsmPrinter &p,
                                  DynTypeParamsConstraintAttr constraint) {
  auto typeName = constraint.getTypeName();
  p << typeName;

  auto paramConstraints = constraint.getParamConstraints();

  p << "<";
  llvm::interleaveComma(paramConstraints, p,
                        [&p](Attribute a) { printTypeConstraint(p, a); });
  p << ">";
}

/// Parse a type constraint.
/// The verifier ensures that the format is respected.
ParseResult parseTypeConstraint(OpAsmParser &p, Attribute *typeConstraint) {
  auto loc = p.getCurrentLocation();

  // Parse an Any constraint.
  auto anyRes = parseOptionalAnyTypeConstraint(p, typeConstraint);
  if (anyRes.has_value())
    return *anyRes;

  // Parse an AnyOf constraint.
  auto anyOfRes = parseOptionalAnyOfTypeConstraint(p, typeConstraint);
  if (anyOfRes.has_value())
    return *anyOfRes;

  // Parse an And constraint.
  auto andRes = parseOptionalAndTypeConstraint(p, typeConstraint);
  if (andRes.has_value())
    return *andRes;

  auto ctx = p.getBuilder().getContext();

  // Type equality constraint.
  // It has the format 'type'.
  Type type;
  auto typeParsed = p.parseOptionalType(type);
  if (typeParsed.has_value()) {
    if (failed(typeParsed.value()))
      return failure();

    *typeConstraint = EqTypeConstraintAttr::get(ctx, type);
    return success();
  }

  if (succeeded(p.parseOptionalQuestion())) {
    StringRef keyword;
    if (failed(p.parseKeyword(&keyword)))
      return failure();
    *typeConstraint = VarTypeConstraintAttr::get(ctx, keyword);
    return success();
  }

  StringRef keyword;
  if (succeeded(p.parseOptionalKeyword(&keyword))) {
    // Parse a non-dynamic type parameter constraint.
    auto irdl = ctx->getOrLoadDialect<IRDLDialect>();
    auto typeWrapper = irdl->getTypeWrapper(keyword);

    if (p.parseLess().succeeded()) {
      // Parse a C++-defined type parameter constraint.
      if (typeWrapper)
        return parseTypeParamsConstraint(p, typeWrapper, typeConstraint);

      // Parse a dynamic type parameter constraint.
      auto paramRes =
          parseOptionalDynTypeParamsConstraint(p, keyword, typeConstraint);
      if (paramRes.has_value())
        return *paramRes;
      p.emitError(loc, "type constraint expected");
      return failure();
    }

    if (typeWrapper) {
      *typeConstraint = TypeBaseConstraintAttr::get(ctx, typeWrapper);
      return success();
    }

    auto baseRes =
        parseOptionalDynTypeBaseConstraint(p, keyword, typeConstraint);
    if (baseRes.has_value())
      return *baseRes;

    p.emitError(loc, "type constraint expected");
    return failure();
  }

  p.emitError(loc, "type constraint expected");
  return failure();
}

/// Print a type constraint.
void printTypeConstraint(OpAsmPrinter &p, Attribute typeConstraint) {
  if (auto eqConstr = typeConstraint.dyn_cast<EqTypeConstraintAttr>()) {
    p << eqConstr.getType();
  } else if (auto anyConstr =
                 typeConstraint.dyn_cast<AnyTypeConstraintAttr>()) {
    p << "Any";
  } else if (auto anyOfConstr =
                 typeConstraint.dyn_cast<AnyOfTypeConstraintAttr>()) {
    printAnyOfTypeConstraint(p, anyOfConstr);
  } else if (auto andConstr =
                 typeConstraint.dyn_cast<AndTypeConstraintAttr>()) {
    printAndTypeConstraint(p, andConstr);
  } else if (auto typeParamsConstr =
                 typeConstraint.dyn_cast<TypeParamsConstraintAttr>()) {
    printTypeParamsConstraint(p, typeParamsConstr);
  } else if (auto typeBaseConstr =
                 typeConstraint.dyn_cast<TypeBaseConstraintAttr>()) {
    p << typeBaseConstr.getTypeDef()->getName();
  } else if (auto dynTypeBaseConstr =
                 typeConstraint.dyn_cast<DynTypeBaseConstraintAttr>()) {
    printDynTypeBaseConstraint(p, dynTypeBaseConstr);
  } else if (auto dynTypeParamsConstr =
                 typeConstraint.dyn_cast<DynTypeParamsConstraintAttr>()) {
    printDynTypeParamsConstraint(p, dynTypeParamsConstr);
  } else if (auto typeConstraintParam =
                 typeConstraint.dyn_cast<VarTypeConstraintAttr>()) {
    p << "?" << typeConstraintParam.getName();
  } else {
    assert(false && "Unknown type constraint.");
  }
}
} // namespace

//===----------------------------------------------------------------------===//
// irdl::DialectOp
//===----------------------------------------------------------------------===//

LogicalResult DialectOp::verify() {
  return success(Dialect::isValidNamespace(name()));
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
  if (failed(parseTypeConstraint(p, &attr)))
    return failure();
  param = NamedTypeConstraintAttr::get(p.getContext(), name, attr);
  return success();
}

void printNamedTypeConstraint(OpAsmPrinter &p, NamedTypeConstraintAttr attr) {
  p.printKeywordOrString(attr.getName());
  p << ": ";
  printTypeConstraint(p, attr.getConstraint());
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
// IRDL operations.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dyn/Dialect/IRDL/IR/IRDLOps.cpp.inc"

//===----------------------------------------------------------------------===//
// IRDL interfaces.
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDLInterfaces.cpp.inc"
