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

} // namespace irdl
} // namespace mlir

namespace mlir {
namespace irdl {
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

  friend llvm::hash_code hash_value(mlir::irdl::OpTypeDef typeDef);
};

inline ArgDefs argDefAllocator(mlir::AttributeStorageAllocator &allocator,
                               ArgDefs argDefs) {
  OwningArgDefs ownArgDefs;
  for (auto &p : argDefs)
    ownArgDefs.emplace_back(allocator.copyInto(p.first), p.second);
  return allocator.copyInto(ArgDefs(ownArgDefs));
}

inline OpTypeDef opTypeDefAllocator(mlir::AttributeStorageAllocator &allocator,
                                    OpTypeDef typeDef) {
  auto allocatedOperandDefs = argDefAllocator(allocator, typeDef.operandDef);
  auto allocatedResultDefs = argDefAllocator(allocator, typeDef.resultDef);
  auto allocatedTraitDefs = allocator.copyInto(typeDef.traitDefs);
  auto allocatedInterfaceDefs = allocator.copyInto(typeDef.interfaceDefs);

  return {allocatedOperandDefs, allocatedResultDefs, allocatedTraitDefs,
          allocatedInterfaceDefs};
}

} // namespace irdl
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h.inc"

#endif // DYN_DIALECT_IRDL_IR_IRDL_H_
