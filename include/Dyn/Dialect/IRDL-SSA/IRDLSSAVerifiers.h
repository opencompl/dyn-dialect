//===- IRDLSSAVerifiers.h - IRDL-SSA verifiers -------------------- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verifiers for objects declared by IRDL-SSA.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_IRDL_SSA_IR_IRDLSSAREGISTRATION_H
#define DYN_IRDL_SSA_IR_IRDLSSAREGISTRATION_H

#include "Dyn/Dialect/IRDL/TypeWrapper.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace irdlssa {

class TypeConstraint;

/// Provides context to the verification of type constraints.
/// Short-lived context for the analysis of a specific use
/// of an IRDL-SSA declaration.
class ConstraintVerifier {
  ArrayRef<std::unique_ptr<TypeConstraint>> constraints;
  SmallVector<Optional<Type>> assigned;

public:
  ConstraintVerifier(ArrayRef<std::unique_ptr<TypeConstraint>> constraints);

  LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             size_t variable);
};

/// A generic type constraint.
class TypeConstraint {
public:
  /// Check that a type is satisfying the type constraint, based
  /// on information fetchable using the context ConstraintVerifier.
  virtual LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ConstraintVerifier &context) const = 0;
};

class IsTypeConstraint : public TypeConstraint {
  Type expectedType;

public:
  IsTypeConstraint(Type expectedType) : expectedType(expectedType) {}

  LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ConstraintVerifier &context) const override;
};

class ParametricTypeConstraint : public TypeConstraint {
  ::mlir::irdl::TypeWrapper *expectedType;
  SmallVector<size_t> constraints;

public:
  ParametricTypeConstraint(::mlir::irdl::TypeWrapper *expectedType,
                           SmallVector<size_t> constraints)
      : expectedType(expectedType), constraints(std::move(constraints)) {}

  LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ConstraintVerifier &context) const override;
};

class DynParametricTypeConstraint : public TypeConstraint {
  DynamicTypeDefinition *expectedType;
  SmallVector<size_t> constraints;

public:
  DynParametricTypeConstraint(DynamicTypeDefinition *expectedType,
                              SmallVector<size_t> constraints)
      : expectedType(expectedType), constraints(std::move(constraints)) {}

  LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ConstraintVerifier &context) const override;
};

class AnyOfTypeConstraint : public TypeConstraint {
  SmallVector<size_t> constraints;

public:
  AnyOfTypeConstraint(SmallVector<size_t> constraints)
      : constraints(std::move(constraints)) {}

  LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ConstraintVerifier &context) const override;
};

class AnyTypeConstraint : public TypeConstraint {
public:
  LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ConstraintVerifier &context) const override;
};

} // namespace irdlssa
} // namespace mlir

#endif // DYN_IRDL_SSA_IR_IRDLSSAREGISTRATION_H
