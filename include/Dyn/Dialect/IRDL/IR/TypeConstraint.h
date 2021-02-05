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

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Hashing.h"

namespace mlir {

namespace dyn {
// Forward declaration.
class DynamicContext;
} // namespace dyn

namespace irdl {

/// A type constraint is for now only a type equality constraint represented by
/// the type name.
struct TypeConstraint {
  std::string typeName;

  bool operator==(const TypeConstraint &o) const {
    return typeName == o.typeName;
  }

  LogicalResult verifyType(Operation *op, Type type, bool isOperand,
                           unsigned pos, dyn::DynamicContext &ctx);
};
} // namespace irdl
} // namespace mlir

#endif // DYN_DIALECT_IRDL_IR_TYPECONSTRAINT_H_
