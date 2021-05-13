//===- DynamicTrait.h - Dynamic trait ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Allow the use of interfaces in dynamic operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICINTERFACE_H
#define DYN_DYNAMICINTERFACE_H

#include "Dyn/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "Dyn/DynamicObject.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace dyn {

// Forward declaration.
class DynamicOpInterfaceImpl;

// In MLIR, interface implementations are stored in an InterfaceTy::Concept
// inside the InterfaceMap stored by the AbstractOperation. The Concept is
// simply a struct containing a pointer to each function an operation has to
// implement.
// In order to have implementations depending on runtime information, we store
// the dynamic information inside class deriving DynamicOpInterfaceImpl. Then,
// we store in the InterfaceMap a function pointer which upon execution will
// retrieve the dynamic operation, and get the interface implementation.
//
// In order to use a c++-defined interface in dynamic operations, the user need
// to derive the DynamicOpInterface and the DynamicOpInterfaceImpl.

/// A dynamic operation interface.
/// Each instance of this class represent a different operation interface.
class DynamicOpInterface : public DynamicObject {
public:
  DynamicOpInterface(DynamicContext *ctx, StringRef name)
      : DynamicObject(ctx), name{name} {}

  /// Give a custom TypeID to the interface.
  /// This should only be used when creating a wrapper around an already defined
  /// interface.
  DynamicOpInterface(DynamicContext *ctx, TypeID id, StringRef name)
      : DynamicObject(ctx, id), name{name} {}

  /// Entry for the interfaceMap. The object should be allocated with malloc,
  /// and the ownership is given to the caller.
  /// The returned object should contain the operation implementation of the
  /// interface.
  virtual void *getConcept() = 0;

  /// Get the interface implementation of an operation.
  /// The operation must be a dynamic operation.
  const DynamicOpInterfaceImpl &getImpl(Operation *op) const;

  const std::string name;
};

/// An implementation of a dynamic interface.
/// This object should contain the information specific to the interface
/// implementation of an operation.
class DynamicOpInterfaceImpl {
public:
  DynamicOpInterfaceImpl(DynamicOpInterface *interface)
      : interface{interface} {}

  DynamicOpInterface *getInterface() { return interface; }

private:
  DynamicOpInterface *interface;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICINTERFACE_H
