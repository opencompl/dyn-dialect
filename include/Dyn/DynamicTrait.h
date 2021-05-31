//===- DynamicTrait.h - Dynamic trait ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Allow the use of traits in dynamic operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DYNAMICTRAIT_H
#define DYN_DYNAMICTRAIT_H

#include "Dyn/DynamicObject.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace dyn {

using OpTraitVerifierFn = std::function<LogicalResult(Operation *op)>;

/// Trait that can be defined dynamically.
class DynamicOpTrait : public DynamicObject {

private:
  DynamicOpTrait(DynamicContext *ctx, StringRef name,
                 OpTraitVerifierFn verifier, TypeID id)
      : DynamicObject(ctx, id), name{name}, verifier(std::move(verifier)) {}

public:
  DynamicOpTrait(DynamicContext *ctx, StringRef name,
                 OpTraitVerifierFn verifier)
      : DynamicObject(ctx), name{name}, verifier(std::move(verifier)) {}

  template <template <typename ConcreteT> class TraitTy>
  static std::unique_ptr<DynamicOpTrait> get(DynamicContext *ctx,
                                             StringRef name) {
    return std::unique_ptr<DynamicOpTrait>(new DynamicOpTrait(
        ctx, name, TraitTy<Operation>::verifyTrait, TypeID::get<TraitTy>()));
  }

  /// Check that the operatio satisfies the trait.
  LogicalResult verifyTrait(Operation *op) { return verifier(op); }

  const std::string name;

private:
  /// Lambda used for checking that the operation satisfy the trait.
  OpTraitVerifierFn verifier;
};

} // namespace dyn
} // namespace mlir

#endif // DYN_DYNAMICTRAIT_H
