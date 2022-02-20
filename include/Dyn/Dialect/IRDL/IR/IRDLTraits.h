//===- IRDLTraits.h - IRDL traits definition ---------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the traits used by the IR Definition Language dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace OpTrait {

/// This class adds the property that there is at most one child of a given op
/// in the operation region. It also provides an API to retrieve the operation.
template <typename... ChildOp> class AtMostOneChildOf {
public:
  template <typename ConcreteType>
  class Impl
      : public ::mlir::OpTrait::TraitBase<ConcreteType,
                                          AtMostOneChildOf<ChildOp...>::Impl> {
  public:
    static LogicalResult verifyTrait(Operation *op) {
      static_assert(
          ConcreteType::template hasTrait<::mlir::OpTrait::OneRegion>(),
          "expected operation to have a single region");

      auto ops = {cast<ConcreteType>(op).template getOps<ChildOp>()...};
      for (auto op : ops) {
        if (!op.empty() && ++op.begin() != op.end()) {
          // TODO: Write a proper error message here.
          return failure();
        }
      }
      return success();
    }

    /// Get the unique operation of a specific op that is in the operation
    /// region.
    template <typename OpT>
    std::enable_if_t<llvm::disjunction<std::is_same<OpT, ChildOp>...>::value,
                     llvm::Optional<OpT>>
    getOp() {
      auto ops =
          cast<ConcreteType>(this->getOperation()).template getOps<OpT>();
      if (ops.empty())
        return {};
      return {*ops.begin()};
    }
  };
};
} // namespace OpTrait
} // namespace mlir
