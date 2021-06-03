//===- IRDL.h - IR Definition Language dialect ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the dialect for the IR Definition Language.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_IRDL_H_
#define DYN_DIALECT_IRDL_IR_IRDL_H_

#include "Dyn/Dialect/IRDL/IR/IRDLInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include <memory>

// Forward declaration.
namespace mlir {
namespace irdl {
class OpTypeDef;
class OpTypeDefAttr;
} // namespace irdl
} // namespace mlir

//===----------------------------------------------------------------------===//
// IRDL Dialect
//===----------------------------------------------------------------------===//

#include "Dyn/Dialect/IRDL/IR/IRDLDialect.h.inc"

namespace mlir {
namespace irdl {
template <typename InterfaceParser>
LogicalResult IRDLDialect::registerOpInterfaceImplParser() {
  auto registered = opInterfaceImplParsers.try_emplace(
      InterfaceParser::Interface::getInterfaceNamespace(),
      new InterfaceParser());
  if (!registered.second)
    return failure();
  return success();
}
} // namespace irdl
} // namespace mlir

//===----------------------------------------------------------------------===//
// IRDL Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dyn/Dialect/IRDL/IR/IRDLOps.h.inc"

#endif // DYN_DIALECT_IRDL_IR_IRDL_H_
