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
    IRDL presents a high-level abstraction for the definition of dialects. It
    can represent dialects, types, and operations.

    The IRDL programs can be used to register dialects at runtime, which enables
    users to plug-in new dialects without having to recompile the project.

    Operation `irdl.dialect` defines a dialect, `irdl.type` defines a type,
    and `irdl.operation` defines an operation.

    A dialect example is shown below:

    ```mlir
    irdl.dialect "cmath" {
      irdl.type "complex"
      irdl.operation "mul"
    }
    ```

    This program defines a `cmath` dialect that defines a `complex` type, and
    a `mul` operation.
  }];

  let extraClassDeclaration = [{
  private:
    /// Register the attributes used by default by IRDL.
    void registerAttributes();
    /// Register the attributes used by standard interfaces wrappers in IRDL.
    void registerStandardInterfaceAttributes();

  public:
    /// Register new attributes that can be used by IRDL.
    template <typename Attr>
    void addNewAttribute() {
      addAttributes<Attr>();
    }
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

def IRDL_DialectOp : IRDL_Op<"dialect", [
    IsolatedFromAbove, SingleBlockImplicitTerminator<"EndDialectOp">
  ]> {
  let summary = "Define a new dialect";
  let description = [{
    `irdl.dialect` operations defines a dialect.

    Example:

    ```mlir
    irdl.dialect "cmath" {
      ...
    }
    ```

    The above program defines a `cmath` dialect.
  }];

  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let printer = [{ ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let verifier = [{ return ::verify(*this); }];
}

def IRDL_EndDialectOp : IRDL_Op<"end_dialect", [Terminator]> {
  let summary = "Implicit terminator of `irdl.dialect` operation";
}

//===----------------------------------------------------------------------===//
// IRDL Type definition
//===----------------------------------------------------------------------===//

def IRDL_TypeOp : IRDL_Op<"type", [HasParent<"DialectOp">]> {
  let summary = "Define a new type";
  let description = [{
    `irdl.type` defines new types belonging to the previously defined dialect
    using `irdl.dialect`.

    Types defined by `irdl.type` are singleton types.

    Example:

    ```mlir
    irdl.dialect "cmath" {
      irdl.type "complex"
    }
    ```
    
    The above program defines a type `complex` inside the dialect `cmath`.
  }];

  let arguments = (ins StrAttr:$name);
  let printer = [{ ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let extraClassDeclaration = [{
    /// Get the parent dialect operation.
    DialectOp getDialectOp() { return cast<DialectOp>(getOperation()->getParentOp()); };
  }];
}

//===----------------------------------------------------------------------===//
// IRDL Alias definition
//===----------------------------------------------------------------------===//

def IRDL_TypeAliasOp : IRDL_Op<"alias", [HasParent<"DialectOp">]> {
  let summary = "Define a new type alias";
  let description = [{
    `irdl.alias` defines new type aliases belonging to the previously defined
    dialect using `irdl.dialect`.

    Types defined by `irdl.aliases` are singleton types.

    Example:

    ```mlir
    irdl.dialect cmath {
      irdl.alias complex = tuple<f32, f32>
      irdl.operation create_complex(real: f32, imaginary: f32) -> !cmath.complex
    }
    ```

    The above program defines a type alias `complex` inside the dialect `cmath`.
    In the `create_complex` operation, `complex` is replaced by
    `tuple<f32, f32>`.
  }];

  let arguments = (ins StrAttr:$name, TypeAttr:$type);
  let printer = [{ ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let extraClassDeclaration = [{
    /// Get the parent dialect operation.
    DialectOp getDialectOp() { return cast<DialectOp>(getOperation()->getParentOp()); };
  }];
}


//===----------------------------------------------------------------------===//
// IRDL Attribute definition
//===----------------------------------------------------------------------===//

def OpDefAttr : Attr<CPred<"$_self.isa<OpTypeDefAttr>()">, "OperationOp type constraints definition"> {
  let storageType = "OpTypeDefAttr";
  let returnType = "OpTypeDef";
}

//===----------------------------------------------------------------------===//
// IRDL Operation definition
//===----------------------------------------------------------------------===//


def IRDL_OperationOp : IRDL_Op<"operation", [HasParent<"DialectOp">]> {
  let summary = "Define a new operation";
  let description = [{
    `irdl.operation` defines new operations belonging to the previously
    defined dialect using `irdl.dialect`.

    Those operations have no verifiers.

    Example:

    ```mlir
    irdl.dialect "cmath" {
      irdl.operation "mul"
    }
    ```
    
    The above program defines an operation `mul` inside the dialect `cmath`.
  }];

  let arguments = (ins StrAttr:$name, OpDefAttr:$op_def);
  let printer = [{ ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let extraClassDeclaration = [{
    /// Get the parent dialect operation.
    DialectOp getDialectOp() { return cast<DialectOp>(getOperation()->getParentOp()); };
  }];
}

#endif // DYN_DIALECT_IRDL_IR_IRDLOPS