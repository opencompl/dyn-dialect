//===- TypeConstraint.h - IRDL type constraint definition -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the different type constraints an operand or a result can
// have.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_TYPECONSTRAINT_H_
#define DYN_DIALECT_IRDL_IR_TYPECONSTRAINT_H_

#include "Dyn/DynamicType.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Hashing.h"

namespace mlir {

namespace dyn {
// Forward declaration.
class DynamicContext;
} // namespace dyn

namespace irdl {
// Forward declaration.
class OperationOp;

/// A generic type constraint.
class TypeConstraint {
public:
  virtual LogicalResult verifyType(Operation *op, Type type, bool isOperand,
                                   unsigned pos, dyn::DynamicContext &ctx) = 0;
};

//===----------------------------------------------------------------------===//
// Equality type constraint
//===----------------------------------------------------------------------===//

class EqTypeConstraint : public TypeConstraint {
public:
  EqTypeConstraint(Type type) : type(type) {}

  /// Get the type constraint given the type name.
  /// The OperationOp should be the op having the type constraint.
  static FailureOr<EqTypeConstraint> get(StringRef typeName, OperationOp op,
                                         dyn::DynamicContext &ctx);

  /// Check that a type is satisfying the type constraint.
  /// The operation should be the operation having the type constraint.
  /// isOperand is used for the error message, and indicate if the constraint
  /// is on an operand or a resuls, and pos is the position of the
  /// operand/result.
  virtual LogicalResult verifyType(Operation *op, Type argType, bool isOperand,
                                   unsigned pos,
                                   dyn::DynamicContext &ctx) override;

  Type type;
};

} // namespace irdl
} // namespace mlir

#endif // DYN_DIALECT_IRDL_IR_TYPECONSTRAINT_H_
