//===- DynamicObject.h - Dynamic dialect ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICOBJECT_H
#define DYN_DYNAMICOBJECT_H

#include "DynamicID.h"

namespace mlir {
namespace dyn {

class DynamicContext;

/// Represent a dynamic object that can be identified with a TypeID.
/// Here, a TypeID is assigned to an instance, and not a statically known class.
class DynamicObject {
public:
  explicit DynamicObject(DynamicContext *ctx);

  inline DynamicContext *getDynamicContext() const { return ctx; }
  inline DynamicID getTypeID() { return dynamicID; }

private:
  DynamicContext *ctx;
  DynamicID dynamicID;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICOBJECT_H
