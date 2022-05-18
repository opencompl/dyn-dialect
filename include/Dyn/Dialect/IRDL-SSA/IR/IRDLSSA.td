//===- IRDLSSA.td - IR Definition Language Dialect ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRDL-SSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_IRDL_SSA_IR_IRDL_SSA
#define DYN_DIALECT_IRDL_SSA_IR_IRDL_SSA

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IRDL Dialect
//===----------------------------------------------------------------------===//

def IRDL_SSA_Dialect : Dialect {
  let summary = "IR Definition Language SSA Dialect";
  let description = [{
    TODO
  }];

  let extraClassDeclaration = [{
  public:
    /// Register new attributes that can be used by IRDL.
    template <typename Attr>
    void addNewAttribute() {
      addAttributes<Attr>();
    }

    /// Add a concrete type wrapper in IRDL.
    template <typename T>
    void addTypeWrapper() {
         addTypeWrapper(std::make_unique<T>());
    }

    /// Get a type wrapper with the given name.
    /// Returns nullptr if no type wrapper with that name is found.
    ::mlir::irdl::TypeWrapper* getTypeWrapper(StringRef typeName);

    // TODO: Why are those declarations needed?
    /// Parse a type registered to this dialect.
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    /// Print a type registered to this dialect.
    void printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const override;

  private:
    /// Add a type wrapper to IRDL.
    void addTypeWrapper(std::unique_ptr<::mlir::irdl::TypeWrapper> wrapper);

    /// Register the attributes used by default by IRDL.
    //void registerAttributes();
    /// Register the attributes used by standard interfaces wrappers in IRDL.
    //void registerStandardInterfaceAttributes();
    /// Wrapper around types that can be used in IRDL.
    llvm::StringMap<std::unique_ptr<::mlir::irdl::TypeWrapper>> typeWrappers;
  }];

  let name = "irdlssa";
  let cppNamespace = "::mlir::irdlssa";
}

#endif // DYN_DIALECT_IRDL_SSA_IR_IRDL_SSA
