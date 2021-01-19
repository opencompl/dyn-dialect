//===- TypeIDAllocator.h - Allocate TypeID dynamically ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYN_TYPEIDALLOCATOR_H
#define DYN_TYPEIDALLOCATOR_H

#include "mlir/Support/TypeID.h"
#include <memory>
#include <vector>

namespace mlir {
namespace dyn {

/// TypeIDs are used to get an unique identifier for each different class.
/// This is heavily used in MLIR to implement isa/dyn_cast functionality.
/// TypeIDs are assigned to classes in MLIR by creating an object in the heap
/// for each class. Here, we need to create TypeIDs at runtime, so we also
/// allocate memory to ensure uniqueness.
/// Here, we ensure uniqueness only as long as the allocator is still alive.
/// If the allocator is destructed, all TypeIDs it allocated are now invalid.
class TypeIDAllocator {
private:
  /// Following MLIR, we align our TypeIDs by one byte.
  struct alignas(8) Storage {};

public:
  /// Allocate a new TypeID, that is ensured to be unique.
  TypeID allocateID() {
    ids.emplace_back(new Storage());
    return TypeID::getFromOpaquePointer(ids.back().get());
  }

private:
  /// The TypeIDs allocated are the addresses of the different Storage.
  /// Keeping those in memory ensure uniqueness of the TypeIDs.
  std::vector<std::unique_ptr<Storage>> ids;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_TYPEIDALLOCATOR_H
