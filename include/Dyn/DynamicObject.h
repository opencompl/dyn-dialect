//===- DynamicObject.h - Dynamic dialect ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Associate TypeID given at runtime to class instances.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICOBJECT_H
#define DYN_DYNAMICOBJECT_H

#include "TypeIDAllocator.h"

namespace mlir {
namespace dyn {

class DynamicContext;

/// Represent a dynamic object that can be identified with a TypeID.
/// Here, a TypeID is assigned to an instance, instead of a class.
class DynamicObject {
public:
  /// Given a dynamic context, create a new dynamic object.
  /// The typeID is allocated by the DynamicContext TypeID allocator.
  explicit DynamicObject(DynamicContext *ctx);

  /// Create a new dynamic object given an already allocated typeID.
  explicit DynamicObject(DynamicContext *ctx, TypeID id);

  inline DynamicContext *getDynamicContext() const { return ctx; }
  inline TypeID getRuntimeTypeID() { return id; }

private:
  DynamicContext *ctx;
  const TypeID id;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICOBJECT_H
