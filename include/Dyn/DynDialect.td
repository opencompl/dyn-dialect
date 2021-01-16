//===- DynDialect.td - Dyn dialect -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT
#define DYN_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dyn dialect definition.
//===----------------------------------------------------------------------===//

def Dyn_Dialect : Dialect {
    let name = "dyn";
    let summary = "A wrapper for dynamics MLIR dialect.";
    let description = [{
        Wrap dynamic dialects and operations.        
    }];
    let cppNamespace = "::mlir::dyn";
}

//===----------------------------------------------------------------------===//
// Base dyn operation definition.
//===----------------------------------------------------------------------===//

class Dyn_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Dyn_Dialect, mnemonic, traits>;

#endif // DYN_DIALECT
