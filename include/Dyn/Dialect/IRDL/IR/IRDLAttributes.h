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

#define GET_ATTRDEF_CLASSES
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h.inc"

#endif // DYN_DIALECT_IRDL_IR_IRDL_H_
