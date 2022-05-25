//===- RegisterIRDL.h - Register dialects defined in IRDL -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Takes a mlir file that defines dialects using IRDL, and register the dialects
// in a DynamicContext.
//
//===----------------------------------------------------------------------===//

#ifndef DYNOPT_REGISTER_IRDL_H
#define DYNOPT_REGISTER_IRDL_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {

class MLIRContext;

/// Register dialects defined in an IRDL file to a dynamic context.
LogicalResult registerIRDL(StringRef irdlFile, MLIRContext *ctx);

/// Register dialects defined in an IRDL-SSA file to a dynamic context.
LogicalResult registerIRDLSSA(StringRef irdlssaFile, MLIRContext *ctx);

} // namespace mlir

#endif // DYNOPT_REGISTER_IRDL_H
