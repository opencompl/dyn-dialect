//===- IRDLSSAInterfaces.h - IRDL-SSA interfaces definition -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the interfaces for the IRDL-SSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_SSA_IR_IRDLSSAINTERFACES_H_
#define DYN_DIALECT_IRDL_SSA_IR_IRDLSSAINTERFACES_H_

#include "Dyn/Dialect/IRDL-SSA/IRDLSSAVerifiers.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>

//===----------------------------------------------------------------------===//
// IRDL-SSA Dialect Interfaces
//===----------------------------------------------------------------------===//
#include "Dyn/Dialect/IRDL-SSA/IR/IRDLSSAInterfaces.h.inc"

#endif //  DYN_DIALECT_IRDL_SSA_IR_IRDLSSAINTERFACES_H_
