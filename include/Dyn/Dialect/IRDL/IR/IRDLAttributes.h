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

#include "Dyn/Dialect/IRDL/IR/IRDLInterface.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/Hashing.h"

namespace mlir {

namespace dyn {
// Forward declaration.
class DynamicContext;
} // namespace dyn

namespace irdl {

// Forward declaration.
class OperationOp;

namespace detail {
// Forward declaration.
struct StringAttributeStorage;
struct TypeAttributeStorage;
} // namespace detail

/// Definition of an argument. An argument is either an operand or a result.
/// It is represented by a name an a type constraint.
using ArgDef = std::pair<StringRef, Attribute>;
using ArgDefs = ArrayRef<ArgDef>;
using OwningArgDefs = std::vector<ArgDef>;

/// Definition of a dynamic operation type.
/// It contains the definition of every operand and result.
class OpTypeDef {
public:
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
class OpTypeDefAttrStorage : public AttributeStorage {
public:
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
    // Here, we need to put the KeyTy (which is OpTypeDef) inside the allocator.
    // For that, we need to walk through the object.

    // We first need to make sure that all the StringRefs are in the allocator.
    // The StringRefs are the name of the operands and results.
    // We are creating new vectors to represent key.operandDef and
    // key.resultDef because we are modifying the StringRef (because we allocate
    // them somewhere else), and we cannot modify key.
    OwningArgDefs operandDefs, resultDefs;
    for (auto &p : key.operandDef)
      operandDefs.emplace_back(allocator.copyInto(p.first), p.second);
    for (auto &p : key.resultDef)
      resultDefs.emplace_back(allocator.copyInto(p.first), p.second);

    // Then we can put the ArgDefs themselves in the allocator.
    auto allocatedOperandDefs = allocator.copyInto(ArgDefs(operandDefs));
    auto allocatedResultDefs = allocator.copyInto(ArgDefs(resultDefs));

    return new (allocator.allocate<OpTypeDefAttrStorage>())
        OpTypeDefAttrStorage({allocatedOperandDefs, allocatedResultDefs});
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

//===----------------------------------------------------------------------===//
// IRDL Equality type constraint attribute
//===----------------------------------------------------------------------===//

/// Attribute for equality type constraint with a dynamic type. The dynamic type
/// is represented by its name.
class EqDynTypeConstraintAttr : public mlir::Attribute::AttrBase<
                                    EqDynTypeConstraintAttr, mlir::Attribute,
                                    mlir::irdl::detail::StringAttributeStorage,
                                    TypeConstraintAttrInterface::Trait> {
public:
  using Base::Base;

  static EqDynTypeConstraintAttr get(MLIRContext &context, StringRef typeName);

  FailureOr<std::unique_ptr<mlir::irdl::TypeConstraint>>
  getTypeConstraint(OperationOp op, dyn::DynamicContext &ctx);

  StringRef getValue();
};

/// Attribute for equality type constraint.
class EqTypeConstraintAttr
    : public mlir::Attribute::AttrBase<EqTypeConstraintAttr, mlir::Attribute,
                                       mlir::irdl::detail::TypeAttributeStorage,
                                       TypeConstraintAttrInterface::Trait> {
public:
  using Base::Base;

  static EqTypeConstraintAttr get(MLIRContext &context, Type type);

  FailureOr<std::unique_ptr<mlir::irdl::TypeConstraint>>
  getTypeConstraint(OperationOp op, dyn::DynamicContext &ctx);

  Type getValue();
};

} // namespace irdl
} // namespace mlir

#endif // DYN_DIALECT_IRDL_IR_IRDL_H_
