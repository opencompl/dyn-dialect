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
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeSupport.h"

// Forward declaration.
namespace mlir {
namespace irdl {
class OperationOp;
class TypeWrapper;
} // namespace irdl
} // namespace mlir

namespace mlir {
namespace irdl {

//===----------------------------------------------------------------------===//
// Operation definition
//===----------------------------------------------------------------------===//

/// Definition of an argument. An argument is either an operand or a result.
/// It is represented by a name an a type constraint.
using ArgDef = std::pair<StringRef, Attribute>;
using ArgDefs = ArrayRef<ArgDef>;
using OwningArgDefs = llvm::SmallVector<ArgDef>;

/// Definition of a dynamic operation type.
/// It contains the definition of every operand and result.
class OpDef {
public:
  ArgDefs typeConstraintVars, operandDef, resultDef;

  ArgDefs getTypeConstraintVars() const { return typeConstraintVars; }

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

  bool operator==(const OpDef &o) const {
    return o.operandDef == operandDef && o.resultDef == resultDef &&
           o.typeConstraintVars == typeConstraintVars;
  }

  friend llvm::hash_code hash_value(mlir::irdl::OpDef typeDef);
};

inline ArgDefs argDefAllocator(mlir::AttributeStorageAllocator &allocator,
                               ArgDefs argDefs) {
  OwningArgDefs ownArgDefs;
  for (auto &p : argDefs)
    ownArgDefs.emplace_back(allocator.copyInto(p.first), p.second);
  return allocator.copyInto(ArgDefs(ownArgDefs));
}

inline OpDef opDefAllocator(mlir::AttributeStorageAllocator &allocator,
                            OpDef typeDef) {
  auto allocatedTypeConstrVars =
      argDefAllocator(allocator, typeDef.typeConstraintVars);
  auto allocatedOperandDefs = argDefAllocator(allocator, typeDef.operandDef);
  auto allocatedResultDefs = argDefAllocator(allocator, typeDef.resultDef);

  return {allocatedTypeConstrVars, allocatedOperandDefs, allocatedResultDefs};
}

} // namespace irdl
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h.inc"

#endif // DYN_DIALECT_IRDL_IR_IRDL_H_
