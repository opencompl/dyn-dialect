//===- TypeConstraint.h - IRDL type constraint definition -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the different type constraints an operand or a result can
// have.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_TYPECONSTRAINT_H_
#define DYN_DIALECT_IRDL_IR_TYPECONSTRAINT_H_

#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "Dyn/Dialect/IRDL/TypeWrapper.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Hashing.h"

namespace mlir {
namespace irdl {
// Forward declaration.
class OperationOp;

/// A generic type constraint.
class TypeConstraint {
public:
  /// Check that a type is satisfying the type constraint.
  /// typeConstraintVars are the constraints associated to the variables. They
  /// are accessed by their index.
  /// varsValue contains the values of the constraint variables that are already
  /// defined, or contains Type{} if the value is not set yet.
  virtual LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
             MutableArrayRef<Type> varsValue) = 0;
};

//===----------------------------------------------------------------------===//
// Equality type constraint
//===----------------------------------------------------------------------===//

class EqTypeConstraint : public TypeConstraint {
public:
  EqTypeConstraint(Type expectedType) : expectedType(expectedType) {}

  virtual LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
             MutableArrayRef<Type> varsValue) override;

private:
  Type expectedType;
};

//===----------------------------------------------------------------------===//
// AnyOf type constraint
//===----------------------------------------------------------------------===//

/// AnyOf type constraint.
/// A type satisfies this constraint if it is included in a set of types.
class AnyOfTypeConstraint : public TypeConstraint {
public:
  AnyOfTypeConstraint(SmallVector<std::unique_ptr<TypeConstraint>> constrs)
      : constrs(std::move(constrs)) {}

  virtual LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
             MutableArrayRef<Type> varsValue) override;

private:
  llvm::SmallVector<std::unique_ptr<TypeConstraint>> constrs;
};

//===----------------------------------------------------------------------===//
// And type constraint
//===----------------------------------------------------------------------===//

/// And type constraint.
/// A type satisfies this constraint if it satisfies a set of constraints.
class AndTypeConstraint : public TypeConstraint {
public:
  AndTypeConstraint(SmallVector<std::unique_ptr<TypeConstraint>> constrs)
      : constrs(std::move(constrs)) {}

  virtual LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
             MutableArrayRef<Type> varsValue) override;

private:
  llvm::SmallVector<std::unique_ptr<TypeConstraint>> constrs;
};

//===----------------------------------------------------------------------===//
// Always true type constraint
//===----------------------------------------------------------------------===//

/// Always true type constraint.
/// All types satisfy this constraint.
class AnyTypeConstraint : public TypeConstraint {
public:
  AnyTypeConstraint() {}

  virtual LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
             MutableArrayRef<Type> varsValue) override {
    return success();
  };
};

//===----------------------------------------------------------------------===//
// Variable type constraint
//===----------------------------------------------------------------------===//

/// Type constraint variable.
/// All types matching the variable should be equal.The first type
/// matching the variable is the one setting the value.
class VarTypeConstraint : public TypeConstraint {
public:
  VarTypeConstraint(size_t varIndex) : varIndex{varIndex} {}

  virtual LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
             MutableArrayRef<Type> varsValue) override;

private:
  size_t varIndex;
};

//===----------------------------------------------------------------------===//
// Base type constraint
//===----------------------------------------------------------------------===//

/// Type constraint asserting that the base type is of a certain dynamic type.
class DynTypeBaseConstraint : public TypeConstraint {
public:
  DynTypeBaseConstraint(DynamicTypeDefinition *dynTypeDef)
      : dynTypeDef(dynTypeDef) {}

  virtual LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
             MutableArrayRef<Type> varsValue) override;

private:
  DynamicTypeDefinition *dynTypeDef;
};

//===----------------------------------------------------------------------===//
// Parameters type constraint
//===----------------------------------------------------------------------===//

/// Type constraint having constraints on dynamic type parameters.
/// A type satisfies this constraint if it has the right expected type,
/// and if each of its parameter satisfies their associated constraint.
class DynTypeParamsConstraint : public TypeConstraint {
public:
  DynTypeParamsConstraint(
      DynamicTypeDefinition *dynTypeDef,
      llvm::SmallVector<std::unique_ptr<TypeConstraint>> &&paramConstraints)
      : dynTypeDef(dynTypeDef), paramConstraints(std::move(paramConstraints)) {}

  virtual LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
             MutableArrayRef<Type> varsValue) override;

private:
  /// TypeID of the parametric type that satisfies this constraint.
  DynamicTypeDefinition *dynTypeDef;

  /// Type constraints of the type parameters.
  llvm::SmallVector<std::unique_ptr<TypeConstraint>> paramConstraints;
};

/// Type constraint having constraints on C++-defined type parameters.
/// A type satisfies this constraint if it has the right expected type,
/// and if each of its parameter satisfies their associated constraint.
class TypeParamsConstraint : public TypeConstraint {
public:
  TypeParamsConstraint(
      TypeWrapper *typeDef,
      llvm::SmallVector<std::unique_ptr<TypeConstraint>> &&paramConstraints)
      : typeDef(typeDef), paramConstraints(std::move(paramConstraints)) {}

  virtual LogicalResult
  verifyType(Optional<function_ref<InFlightDiagnostic()>> emitError, Type type,
             ArrayRef<std::unique_ptr<TypeConstraint>> typeConstraintVars,
             MutableArrayRef<Type> varsValue) override;

private:
  /// Base type that satisfies the constraint.
  TypeWrapper *typeDef;

  /// Type constraints of the type parameters.
  llvm::SmallVector<std::unique_ptr<TypeConstraint>> paramConstraints;
};

} // namespace irdl
} // namespace mlir

#endif // DYN_DIALECT_IRDL_IR_TYPECONSTRAINT_H_
