//===- IRDLSSAAttributes.cpp - Attributes for IRDL-SSA ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSAAttributes.h"
#include "Dyn/Dialect/IRDL/TypeWrapper.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace irdlssa {

using namespace irdl;

Attribute ParamTypeInstanceAttr::parse(AsmParser &odsParser, Type odsType) {
  auto ctx = odsParser.getContext();

  std::string base;
  if (failed(odsParser.parseOptionalString(&base))) {
    return Attribute();
  }

  SmallVector<ParamTypeAttrOrAnyAttr> params;
  if (succeeded(odsParser.parseLess())) {
    while (failed(odsParser.parseOptionalGreater())) {
      SMLoc parseLoc = odsParser.getNameLoc();
      Attribute param = ParamTypeAttrOrAnyAttr::parse(odsParser, odsType);
      if (param.isa_and_nonnull<ParamTypeAttrOrAnyAttr>()) {
        params.push_back(param.cast<ParamTypeAttrOrAnyAttr>());
      } else {
        odsParser.emitError(parseLoc).append(
            "attribute parameter is not a type attribute or a parametric type "
            "attribute");
        return Attribute();
      }
    }
  }

  return ParamTypeInstanceAttr::get(ctx, StringAttr::get(ctx, base), params);
}

void ParamTypeInstanceAttr::print(AsmPrinter &odsPrinter) const {
  this->getBase().print(odsPrinter.getStream());
  ArrayRef<ParamTypeAttrOrAnyAttr> params = this->getParams();
  if (params.size() > 0) {
    odsPrinter << '<';
    llvm::interleaveComma(
        params, odsPrinter,
        [&odsPrinter](ParamTypeAttrOrAnyAttr a) { a.print(odsPrinter); });
    odsPrinter << '>';
  }
}

Attribute ParamTypeAttrOrAnyAttr::parse(AsmParser &odsParser, Type odsType) {
  auto ctx = odsParser.getContext();

  SMLoc parseLoc = odsParser.getNameLoc();
  Attribute res = ParamTypeInstanceAttr::parse(odsParser, odsType);
  if (res.isa_and_nonnull<ParamTypeInstanceAttr>()) {
    return ParamTypeAttrOrAnyAttr::get(ctx, res.cast<ParamTypeInstanceAttr>());
  } else if (succeeded(odsParser.parseAttribute(res))) {
    return ParamTypeAttrOrAnyAttr::get(ctx, res);
  } else {
    odsParser.emitError(parseLoc, "failed to parse type parameter attribute");
    return Attribute();
  }
}

void ParamTypeAttrOrAnyAttr::print(AsmPrinter &odsPrinter) const {
  if (ParamTypeInstanceAttr attr =
          this->getAttr().dyn_cast<ParamTypeInstanceAttr>()) {
    attr.print(odsPrinter);
  } else {
    odsPrinter << this->getAttr();
  }
}

Attribute ParamTypeAttrOrAnyAttr::instantiateParamType(
    llvm::function_ref<InFlightDiagnostic()> emitError, MLIRContext &ctx) {
  ParamTypeInstanceAttr typeDesc =
      this->getAttr().dyn_cast<ParamTypeInstanceAttr>();

  if (!typeDesc)
    return this->getAttr();

  auto typeName = typeDesc.getBase();

  SmallVector<Attribute> params;
  for (ParamTypeAttrOrAnyAttr param : typeDesc.getParams()) {
    auto result = param.instantiateParamType(emitError, ctx);
    if (!result)
      return Attribute();

    params.push_back(result);
  }

  if (DynamicTypeDefinition *type = findDynamicType(ctx, typeName)) {
    DynamicType instantiated = DynamicType::getChecked(emitError, type, params);
    if (!instantiated)
      return Attribute();
    return TypeAttr::get(instantiated);
  } else if (TypeWrapper *type = findTypeWrapper(ctx, typeName)) {
    Type instantiated = type->instantiate(emitError, params);
    if (!instantiated)
      return Attribute();
    return TypeAttr::get(instantiated);
  } else {
    emitError().append("type ", typeName, " is not declared at that point");
    return {};
  }
}

} // namespace irdlssa
} // namespace mlir
