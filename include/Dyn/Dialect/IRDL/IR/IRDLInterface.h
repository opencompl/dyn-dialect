//===- IRDLInterface.h - IR Definition Language dialect ------------------*- C++
//-*-===//
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

#ifndef DYN_DIALECT_IRDL_IR_IRDLINTERFACE_H_
#define DYN_DIALECT_IRDL_IR_IRDLINTERFACE_H_

#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <memory>

// Forward declaration.
namespace mlir {
namespace irdl {
class TypeConstraint;
}
namespace dyn {
class DynamicContext;
}
} // namespace mlir

//===----------------------------------------------------------------------===//
// IRDL Dialect Interfaces
//===----------------------------------------------------------------------===//
#include "Dyn/Dialect/IRDL/IR/IRDLInterface.h.inc"

#endif //  DYN_DIALECT_IRDL_IR_IRDLINTERFACE_H_
