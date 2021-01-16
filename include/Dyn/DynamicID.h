//===- DynamicID.h - Unique dynamic identifier ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICID_H
#define DYN_DYNAMICID_H

#include <cassert>
#include <limits>

namespace mlir {
namespace dyn {

/// Unique ID generated dynamically.
/// Uniqueness is ensured between DynamicIDs allocated with the same
/// DynamicIDAllocator.
class DynamicID {
private:
  friend class DynamicIDAllocator;
  DynamicID(std::size_t id) : id(id) {}

  /// Underlying id value
  std::size_t id;
};

/// DynamicID allocator ensuring uniqueness.
class DynamicIDAllocator {
public:
  DynamicID allocateID() {
    assert(nextID != std::numeric_limits<std::size_t>::max() &&
           "Maximum number of dynamic allocations reached");
    return DynamicID(nextID++);
  }

private:
  /// Underlying id value of the next id allocated.
  /// This number should always go up.
  std::size_t nextID{};
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICID_H
