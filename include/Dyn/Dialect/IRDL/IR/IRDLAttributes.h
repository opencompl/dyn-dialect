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

#include "Dyn/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeSupport.h"

namespace mlir {

namespace dyn {
// Forward declaration.
class DynamicContext;
class DynamicOpTrait;
} // namespace dyn

namespace irdl {

// Forward declaration.
class OperationOp;

namespace detail {
// Forward declaration.
struct TypeAttributeStorage;
struct TypeArrayAttrStorage;
} // namespace detail

namespace detail {
/// Attribute storage for string arrays.
/// This should be moved somewhere else in MLIR.
struct StringArrayAttrStorage : public AttributeStorage {
  using KeyTy = ArrayRef<StringRef>;

  StringArrayAttrStorage(ArrayRef<StringRef> values) : values(values) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == values; }

  static StringArrayAttrStorage *construct(AttributeStorageAllocator &allocator,
                                           KeyTy key);

  ArrayRef<StringRef> values;
};
} // namespace detail

/// Definition of an argument. An argument is either an operand or a result.
/// It is represented by a name an a type constraint.
using ArgDef = std::pair<StringRef, Attribute>;
using ArgDefs = ArrayRef<ArgDef>;
using OwningArgDefs = std::vector<ArgDef>;
using TraitDefs = ArrayRef<mlir::dyn::DynamicOpTrait *>;
using OwningTraitDefs = std::vector<mlir::dyn::DynamicOpTrait *>;
using InterfaceDefs = ArrayRef<InterfaceImplAttrInterface>;
using OwningInterfaceDefs = std::vector<InterfaceImplAttrInterface>;

/// Definition of a dynamic operation type.
/// It contains the definition of every operand and result.
class OpTypeDef {
public:
  ArgDefs operandDef, resultDef;
  TraitDefs traitDefs;
  InterfaceDefs interfaceDefs;

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

  /// Return the traits definitions.
  /// A trait is defined by its name.
  TraitDefs getTraitsDefinitions() const { return traitDefs; };

  /// Return the interface definitions.
  InterfaceDefs getInterfaceDefinitions() const { return interfaceDefs; }

  bool operator==(const OpTypeDef &o) const {
    return o.operandDef == operandDef && o.resultDef == resultDef &&
           o.traitDefs == traitDefs && o.interfaceDefs == interfaceDefs;
  }
};

/// Storage for OpTypeDefAttr.
class OpTypeDefAttrStorage : public AttributeStorage {
public:
  using KeyTy = OpTypeDef;

  OpTypeDefAttrStorage(ArgDefs operandDefs, ArgDefs resultDefs,
                       TraitDefs traitDefs, InterfaceDefs interfaceDefs)
      : opTypeDef({operandDefs, resultDefs, traitDefs, interfaceDefs}) {}

  bool operator==(const KeyTy &key) const { return key == opTypeDef; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.operandDef, key.resultDef, key.traitDefs,
                              key.interfaceDefs);
  }

  static KeyTy getKey(ArgDefs operandDefs, ArgDefs resultDefs,
                      TraitDefs traitDefs, InterfaceDefs interfaceDefs) {
    return KeyTy({operandDefs, resultDefs, traitDefs, interfaceDefs});
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
    auto allocatedTraitDefs = allocator.copyInto(key.traitDefs);
    auto allocatedInterfaceDefs = allocator.copyInto(key.interfaceDefs);

    return new (allocator.allocate<OpTypeDefAttrStorage>())
        OpTypeDefAttrStorage({allocatedOperandDefs, allocatedResultDefs,
                              allocatedTraitDefs, allocatedInterfaceDefs});
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
                           ArgDefs resultDefs, TraitDefs traitDefs,
                           InterfaceDefs interfaceDefs) {
    return Base::get(&ctx, operandDefs, resultDefs, traitDefs, interfaceDefs);
  }

  OpTypeDef getValue() { return getImpl()->opTypeDef; }
};

//===----------------------------------------------------------------------===//
// IRDL Equality type constraint attribute
//===----------------------------------------------------------------------===//

/// Attribute for equality type constraint.
class EqTypeConstraintAttr
    : public mlir::Attribute::AttrBase<EqTypeConstraintAttr, mlir::Attribute,
                                       mlir::irdl::detail::TypeAttributeStorage,
                                       TypeConstraintAttrInterface::Trait> {
public:
  using Base::Base;

  static EqTypeConstraintAttr get(MLIRContext &context, Type type);

  std::unique_ptr<mlir::irdl::TypeConstraint>
  getTypeConstraint(dyn::DynamicContext &ctx);

  Type getValue();
};

//===----------------------------------------------------------------------===//
// IRDL AnyOf type constraint attribute
//===----------------------------------------------------------------------===//

/// Attribute for the AnyOf type constraint.
class AnyOfTypeConstraintAttr
    : public mlir::Attribute::AttrBase<AnyOfTypeConstraintAttr, mlir::Attribute,
                                       mlir::irdl::detail::TypeArrayAttrStorage,
                                       TypeConstraintAttrInterface::Trait> {
public:
  using Base::Base;

  static AnyOfTypeConstraintAttr get(MLIRContext &context, ArrayRef<Type> type);

  std::unique_ptr<mlir::irdl::TypeConstraint>
  getTypeConstraint(dyn::DynamicContext &ctx);

  ArrayRef<Type> getValue();
};

//===----------------------------------------------------------------------===//
// Always true type constraint attribute
//===----------------------------------------------------------------------===//

/// Attribute for equality type constraint.
class AnyTypeConstraintAttr
    : public mlir::Attribute::AttrBase<AnyTypeConstraintAttr, mlir::Attribute,
                                       mlir::AttributeStorage,
                                       TypeConstraintAttrInterface::Trait> {
public:
  using Base::Base;

  static AnyTypeConstraintAttr get(MLIRContext &context);

  std::unique_ptr<mlir::irdl::TypeConstraint>
  getTypeConstraint(dyn::DynamicContext &ctx);
};

} // namespace irdl
} // namespace mlir

#endif // DYN_DIALECT_IRDL_IR_IRDL_H_
