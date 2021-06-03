//===- IRDL.td - IR Definition Language Dialect ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRDL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_IR_IRDL
#define DYN_DIALECT_IRDL_IR_IRDL

include "mlir/IR/OpBase.td"

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

  let isExtensible = 1;

  let extraClassDeclaration = [{
  public:
    /// Register new attributes that can be used by IRDL.
    template <typename Attr>
    void addNewAttribute() {
      addAttributes<Attr>();
    }
    /// This field is used during parsing. If this field is non-null, it points
    /// to the dialect that is currently being parsed by MLIR.
    ExtensibleDialect *currentlyParsedDialect = nullptr;

    /// Get an interface implementation parser given its name.
    FailureOr<DynamicOpInterfaceImplParser*> lookupOpInterfaceImplParser(StringRef name) const;

    /// Register a parser for an operation interface implementation.
    /// The parser will be used during the parsing of the interface
    /// implementations during the operation definition parsing.
    template<typename Interface>
    LogicalResult registerOpInterfaceImplParser();

  private:
    /// Register the attributes used by default by IRDL.
    void registerAttributes();
    /// Register the attributes used by standard interfaces wrappers in IRDL.
    void registerStandardInterfaceAttributes();

    llvm::StringMap<std::unique_ptr<mlir::irdl::DynamicOpInterfaceImplParser>> opInterfaceImplParsers;
  }];

  let name = "irdl";
  let cppNamespace = "::mlir::irdl";
}

class IRDL_AttrDef<string name, list<Trait> traits = []>
  : AttrDef<IRDL_Dialect, name, traits, "::mlir::Attribute"> {
  let cppClassName = name;
  let mnemonic = ?;
}

#endif // DYN_DIALECT_IRDL_IR_IRDL
