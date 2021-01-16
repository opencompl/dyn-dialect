//===- DynOps.td - Dyn dialect ops -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYN_OPS
#define DYN_OPS

include "DynDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Dyn_FooOp : Dyn_Op<"foo", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Test operation.";
    let description = [{
        Test operation.
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // DYN_OPS
