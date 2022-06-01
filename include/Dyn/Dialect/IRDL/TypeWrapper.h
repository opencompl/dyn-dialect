//===- TypeWrapper.h - IRDL type wrapper definition -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the type wrapper for IRDL that allows wrapping an
// already-defined type for IRDL use.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_TYPEWRAPPER_H_
#define DYN_DIALECT_IRDL_IR_TYPEWRAPPER_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallString.h"

namespace mlir {
namespace irdl {
/// A wrapper around a C++-defined type to extract type parameters.
/// For most cases, TypeWrapper should be used instead.
class TypeWrapper {
public:
  /// Check if the given type is the one wrapped.
  virtual bool isCorrectType(mlir::Type t) = 0;

  /// Get the parameters of a type as attributes.
  /// The type should be the one wrapped, which is checked with `isCorrectType`.
  virtual llvm::SmallVector<mlir::Attribute> getParameters(mlir::Type t) = 0;

  /// Returns the type name, including the dialect prefix.
  virtual llvm::StringRef getName() = 0;

  /// Returns the amount of parameters the type expects.
  virtual size_t getParameterAmount() = 0;
};

/// A wrapper around a concrete C++-defined type.
template <typename T> class ConcreteTypeWrapper : public TypeWrapper {
public:
  /// Get the parameters of a type as attributes.
  virtual llvm::SmallVector<mlir::Attribute> getParameters(T t) = 0;

  llvm::SmallVector<mlir::Attribute> getParameters(mlir::Type type) override {
    return getParameters(type.cast<T>());
  };

  bool isCorrectType(mlir::Type type) override { return type.isa<T>(); }
};

/// A wrapper around a dynamic type.
class DynamicTypeWrapper : public TypeWrapper {
  DynamicTypeDefinition *dynType;
  size_t parameterAmount;
  SmallString<32> completeName;

public:
  DynamicTypeWrapper(SmallString<32> completeName, DynamicTypeDefinition *dynType,
                     size_t parameterAmount)
      : dynType(dynType), parameterAmount(parameterAmount),
        completeName(completeName) {}

  bool isCorrectType(mlir::Type t) override {
    auto dynType = t.dyn_cast<DynamicType>();
    return dynType && dynType.getTypeDef() == this->dynType;
  }

  llvm::SmallVector<mlir::Attribute> getParameters(mlir::Type t) override {
    auto params = t.cast<DynamicType>().getParams();
    return SmallVector<Attribute>(params.begin(), params.end());
  }

  llvm::StringRef getName() override { return this->completeName; }

  size_t getParameterAmount() override { return this->parameterAmount; }
};

} // namespace irdl
} // namespace mlir

#endif // DYN_DIALECT_IRDL_IR_TYPEWRAPPER_H_
