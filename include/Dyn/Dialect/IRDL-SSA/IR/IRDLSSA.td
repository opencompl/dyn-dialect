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
    IRDL-SSA is an SSA-based backend for IRDL. It allows encoding IRDL
    dialect constraints in SSA form, allowing easier analysis and
    optimization.

    IRDL-SSA code is meant to be generated and not written by hand.
    As such, the design focuses on ease of generation/analysis instead
    of ease of writing/reading.

    Operation `irdlssa.dialect` defines a dialect, `irdlssa.type` defines
    a type and `irdlssa.operation` defines an operation. Constraints on
    type variables are also expressed using operations.

    A dialect example is shown below:

    ```mlir
    irdlssa.dialect cmath {
      irdlssa.type complex {
        %0 = irdlssa.is_type : f32
        %1 = irdlssa.is_type : f64
        %2 = irdlssa.any_of(%0, %1)
        irdlssa.parameters(%2)
      }

      irdlssa.operation mul {
        %0 = irdlssa.is_type : f32
        %1 = irdlssa.is_type : f64
        %2 = irdlssa.any_of(%0, %1)
        %3 = irdlssa.parametric_type : "cmath.complex"<%2>
        irdlssa.operands(%3, %3)
        irdlssa.results(%3)
      }
    }
    ```

    This program defines a `cmath` dialect that defines a `complex` type, and
    a `mul` operation. Both express constraints over their parameters using
    SSA constraint operations. Informally, one can see those SSA values as
    constraint variables that evaluate to a single type at constraint
    evaluation. For example, the result of the `irdlssa.any_of` stored in `%2`
    in the `mul` operation will collapse into either `f32` or `f64` for the
    entirety of this instance of `mul` constraint evaluation. As such,
    both operands and the result of `mul` must be of equal type (and not just
    satisfy the same constraint).
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
