//===- IRDLInterfaces.h - IRDL interfaces definition ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the interfaces for the IR Definition Language dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_IRDLINTERFACES_H_
#define DYN_DIALECT_IRDL_IR_IRDLINTERFACES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Dyn/Dialect/IRDL/IRDLContext.h"
#include <memory>

// Forward declaration.
namespace mlir {
namespace irdl {
class TypeConstraint;
}
} // namespace mlir

//===----------------------------------------------------------------------===//
// IRDL Dialect Interfaces
//===----------------------------------------------------------------------===//
#include "Dyn/Dialect/IRDL/IR/IRDLInterfaces.h.inc"

#endif //  DYN_DIALECT_IRDL_IR_IRDLINTERFACES_H_
