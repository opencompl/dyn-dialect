//===- IRDLInterface.h - Interfaces defined in IRDL -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines interfaces that can be implemented by IRDL-defined operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_IRDLINTERFACE_H
#define DYN_IRDLINTERFACE_H

#include "Dyn/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"

// Forward declaration
namespace mlir {
namespace dyn {
class DynamicOpInterface;
} // namespace dyn
namespace irdl {
class InterfaceImplAttrInterface;
}
} // namespace mlir

namespace mlir {
namespace irdl {

/// Format for a DynamicOpInterfaceImpl.
/// It is used in IRDL to parse interface implementations for operations.
class DynamicOpInterfaceImplParser {
public:
  virtual ParseResult parseImpl(OpAsmParser &p,
                                InterfaceImplAttrInterface &interface) = 0;
};

} // namespace irdl
} // namespace mlir

#endif // DYN_IRDLINTERFACE_H
