//===- DynamicType.cpp - Dynamic types --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represent types that can be defined at runtime.
//
//===----------------------------------------------------------------------===//

#include "Dyn/DynamicType.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace dyn;

DynamicTypeDefinition::DynamicTypeDefinition(Dialect *dialect,
                                             llvm::StringRef name)
    : DynamicObject(dialect->getContext()->getLoadedDialect<DynamicContext>()),
      dialect(dialect), name(name.str()) {}

DynamicTypeDefinition *DynamicType::getTypeDef() { return getImpl()->typeDef; }

DynamicType DynamicType::get(MLIRContext *ctx, DynamicTypeDefinition *typeDef) {
  return detail::TypeUniquer::getWithTypeID<DynamicType>(
      ctx, typeDef->getRuntimeTypeID(), typeDef);
}
