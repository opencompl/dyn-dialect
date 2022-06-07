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

#ifndef DYN_DIALECT_IRDL_SSA_IR_IRDLSSAATTRIBUTES_H_
#define DYN_DIALECT_IRDL_SSA_IR_IRDLSSAATTRIBUTES_H_

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeSupport.h"

// Forward declarations
namespace mlir {
namespace irdlssa {
class TypeOrParamTypeAttr;
}
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSAAttributes.h.inc"

#endif // DYN_DIALECT_IRDL_SSA_IR_IRDLSSAATTRIBUTES_H_
