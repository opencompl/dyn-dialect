module {
  irdl.dialect @arm_sve {
    irdl.operation @intr.fadd {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @masked.addf {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%2, %5, %8)
      irdl.results(%11)
    }
    irdl.operation @intr.add {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @masked.addi {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%2, %5, %8)
      irdl.results(%11)
    }
    irdl.operation @intr.fdiv {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @masked.divf {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%2, %5, %8)
      irdl.results(%11)
    }
    irdl.operation @intr.fmul {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @masked.mulf {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%2, %5, %8)
      irdl.results(%11)
    }
    irdl.operation @intr.mul {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @masked.muli {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%2, %5, %8)
      irdl.results(%11)
    }
    irdl.operation @intr.sdiv {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @masked.divi_signed {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%2, %5, %8)
      irdl.results(%11)
    }
    irdl.operation @intr.fsub {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @masked.subf {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%2, %5, %8)
      irdl.results(%11)
    }
    irdl.operation @intr.sub {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @masked.subi {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%2, %5, %8)
      irdl.results(%11)
    }
    irdl.operation @intr.udiv {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @masked.divi_unsigned {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%2, %5, %8)
      irdl.results(%11)
    }
    irdl.operation @intr.sdot {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @sdot {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %6 = irdl.any_of(%4, %5) 
      %7 = irdl.all_of(%3, %6) 
      %8 = irdl.all_of(%2, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.any_of(%13, %14) 
      %16 = irdl.all_of(%12, %15) 
      %17 = irdl.all_of(%11, %16) 
      %18 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.any_of(%22, %23) 
      %25 = irdl.all_of(%21, %24) 
      %26 = irdl.all_of(%20, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %33 = irdl.any_of(%31, %32) 
      %34 = irdl.all_of(%30, %33) 
      %35 = irdl.all_of(%29, %34) 
      irdl.operands(%8, %17, %26)
      irdl.results(%35)
    }
    irdl.operation @intr.smmla {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @smmla {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %5 = irdl.any_of(%4) 
      %6 = irdl.all_of(%3, %5) 
      %7 = irdl.all_of(%2, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %9 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %13 = irdl.any_of(%12) 
      %14 = irdl.all_of(%11, %13) 
      %15 = irdl.all_of(%10, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %17 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%18, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %25 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %29 = irdl.any_of(%28) 
      %30 = irdl.all_of(%27, %29) 
      %31 = irdl.all_of(%26, %30) 
      irdl.operands(%7, %15, %23)
      irdl.results(%31)
    }
    irdl.operation @intr.udot {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @udot {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %6 = irdl.any_of(%4, %5) 
      %7 = irdl.all_of(%3, %6) 
      %8 = irdl.all_of(%2, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.any_of(%13, %14) 
      %16 = irdl.all_of(%12, %15) 
      %17 = irdl.all_of(%11, %16) 
      %18 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.any_of(%22, %23) 
      %25 = irdl.all_of(%21, %24) 
      %26 = irdl.all_of(%20, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %33 = irdl.any_of(%31, %32) 
      %34 = irdl.all_of(%30, %33) 
      %35 = irdl.all_of(%29, %34) 
      irdl.operands(%8, %17, %26)
      irdl.results(%35)
    }
    irdl.operation @intr.ummla {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8)
      irdl.results(%9)
    }
    irdl.operation @ummla {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %5 = irdl.any_of(%4) 
      %6 = irdl.all_of(%3, %5) 
      %7 = irdl.all_of(%2, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %9 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %13 = irdl.any_of(%12) 
      %14 = irdl.all_of(%11, %13) 
      %15 = irdl.all_of(%10, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %17 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%18, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %25 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %29 = irdl.any_of(%28) 
      %30 = irdl.all_of(%27, %29) 
      %31 = irdl.all_of(%26, %30) 
      irdl.operands(%7, %15, %23)
      irdl.results(%31)
    }
  }
}
