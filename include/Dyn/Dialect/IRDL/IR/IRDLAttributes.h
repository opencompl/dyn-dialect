//===- IRDLAttributes.h - Attributes definition for IRDL --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes used in the IRDL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_IRDLATTRIBUTES_H_
#define DYN_DIALECT_IRDL_IR_IRDLATTRIBUTES_H_

#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "Dyn/DynamicObject.h"
#include "Dyn/DynamicType.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/Hashing.h"

namespace mlir {
namespace irdl {

/// A type constraint is for now only a type equality constraint represented by
/// the type name.
struct TypeConstraint {
  std::string typeName;

  bool operator==(const TypeConstraint &o) const {
    return typeName == o.typeName;
  }

  LogicalResult verifyType(Operation *op, Type type, bool isOperand,
                           unsigned pos, dyn::DynamicContext &ctx) {
    auto dialectEnd = typeName.find('.');
    assert(dialectEnd != std::string::npos);
    auto dialectName = StringRef(typeName).substr(0, dialectEnd);
    auto typeSubname = StringRef(typeName).substr(dialectEnd + 1);

    auto dialectRes = ctx.lookupDialect(dialectName);
    if (failed(dialectRes))
      return op->emitError("dialect " + dialectName + " is not registered.");
    auto *dialect = *dialectRes;

    auto dynTypeRes = dialect->lookupType(typeSubname);
    if (failed(dynTypeRes))
      return op->emitError("type " + typeSubname +
                           " is not registered in the dialect " + dialectName +
                           ".");
    auto *dynType = *dynTypeRes;

    if (!dyn::DynamicType::isa(type, dynType)) {
      auto argType = isOperand ? "operand" : "result";
      return op->emitError(std::to_string(pos) + "nth " + argType +
                           " should be of type " + typeName);
    }

    return success();
  }
};

/// Make type constraints hashable.
inline llvm::hash_code hash_value(const TypeConstraint &constr) {
  return llvm::hash_value(constr.typeName);
}

/// Definition of an argument. An argument is either an operand or a result.
/// It is represented by a name an a type constraint.
using ArgDef = std::pair<std::string, TypeConstraint>;
using ArgDefs = ArrayRef<ArgDef>;
using OwningArgDefs = std::vector<ArgDef>;

/// Definition of a dynamic operation type.
/// It contains the definition of every operand and result.
struct OpTypeDef {
  ArgDefs operandDef, resultDef;

  /// Get the number of operands.
  std::size_t getNumOperands() const { return operandDef.size(); }

  /// Return the operand definitions.
  /// Each operand is defined by a name, and a type constraint.
  ArgDefs getOperandDefinitions() const { return operandDef; }

  /// Get the number of results.
  std::size_t getNumResults() const { return resultDef.size(); }

  /// Return the result definitions.
  /// Each result is defined by a name, and a type constraint.
  ArgDefs getResDefinitions() const { return resultDef; }

  bool operator==(const OpTypeDef &o) const {
    return o.operandDef == operandDef && o.resultDef == o.resultDef;
  }
};

/// Storage for OpTypeDefAttr.
struct OpTypeDefAttrStorage : public AttributeStorage {
  using KeyTy = OpTypeDef;

  OpTypeDefAttrStorage(ArgDefs operandDefs, ArgDefs resultDefs)
      : opTypeDef({operandDefs, resultDefs}) {}

  bool operator==(const KeyTy &key) const { return key == opTypeDef; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.operandDef, key.resultDef);
  }

  static KeyTy getKey(ArgDefs operandDefs, ArgDefs resultDefs) {
    return KeyTy({operandDefs, resultDefs});
  }

  static OpTypeDefAttrStorage *
  construct(mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    auto operandDefs = allocator.copyInto(key.operandDef);
    auto resDefs = allocator.copyInto(key.resultDef);

    return new (allocator.allocate<OpTypeDef>())
        OpTypeDefAttrStorage({operandDefs, resDefs});
  }

  OpTypeDef opTypeDef;
};

/// Attribute representing the type definition of a dynamic operation.
/// It contains a name and type constraints for each operand and result.
class OpTypeDefAttr
    : public mlir::Attribute::AttrBase<OpTypeDefAttr, mlir::Attribute,
                                       OpTypeDefAttrStorage> {
public:
  /// Using Attribute constructors.
  using Base::Base;

  static OpTypeDefAttr get(MLIRContext &ctx, ArgDefs operandDefs,
                           ArgDefs resultDefs) {
    return Base::get(&ctx, operandDefs, resultDefs);
  }

  OpTypeDef getValue() { return getImpl()->opTypeDef; }
};

} // namespace irdl
} // namespace mlir

#endif // DYN_DIALECT_IRDL_IR_IRDL_H_
