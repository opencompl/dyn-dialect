//===- IRDLOps.td - IR Definition Language Dialect ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRDL dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_IRDLOPS
#define DYN_DIALECT_IRDL_IR_IRDLOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// IRDL Dialect
//===----------------------------------------------------------------------===//

def IRDL_Dialect : Dialect {
  let summary = "IR Definition Language Dialect";
  let description = [{
    TODO
  }];

  let name = "irdl";
  let cppNamespace = "::mlir::irdl";
}

class IRDL_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<IRDL_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// IRDL Dialect definition
//===----------------------------------------------------------------------===//

def IRDL_DialectOp : IRDL_Op<"dialect", [IsolatedFromAbove, SingleBlockImplicitTerminator<"EndDialectOp">]> {
    let summary = "Define a new dialect";
    let description = [{
      TODO
    }];

    let arguments = (ins StrAttr:$name);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "attr-dict $body";
}

def IRDL_EndDialectOp : IRDL_Op<"end_dialect", [Terminator]> {
    let summary = "Define a new dialect";
    let description = [{
      TODO
    }];

    let assemblyFormat = "attr-dict";
}

#endif // DYN_DIALECT_IRDL_IR_IRDLOPS