module {
  irdl.dialect @vector {
    irdl.operation @vscale {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @bitcast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @broadcast {
      %0 = irdl.any
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      irdl.operands(%0)
      irdl.results(%3)
    }
    irdl.operation @compressstore {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %13 = irdl.any_of(%12) 
      %14 = irdl.all_of(%11, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %20 = irdl.any_of(%19) 
      %21 = irdl.all_of(%18, %20) 
      irdl.operands(%2, %3, %15, %21)
      irdl.results()
    }
    irdl.operation @constant_mask {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      irdl.operands()
      irdl.results(%2)
    }
    irdl.operation @contract {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any
      %11 = irdl.any
      irdl.operands(%4, %9, %10)
      irdl.results(%11)
    }
    irdl.operation @create_mask {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      irdl.operands(%0)
      irdl.results(%3)
    }
    irdl.operation @expandload {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %13 = irdl.any_of(%12) 
      %14 = irdl.all_of(%11, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %20 = irdl.any_of(%19) 
      %21 = irdl.all_of(%18, %20) 
      %22 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %24 = irdl.all_of(%22, %23) 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %26 = irdl.any_of(%25) 
      %27 = irdl.all_of(%24, %26) 
      irdl.operands(%2, %3, %15, %21)
      irdl.results(%27)
    }
    irdl.operation @extractelement {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %4 = irdl.any
      irdl.operands(%2, %3)
      irdl.results(%4)
    }
    irdl.operation @extract {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.any
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @extract_strided_slice {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @fma {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%2, %5, %8)
      irdl.results(%11)
    }
    irdl.operation @flat_transpose {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((elementType.isSignedInteger())) || ((::llvm::isa<::mlir::IndexType>(elementType))) || ((::llvm::isa<::mlir::FloatType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((elementType.isSignedInteger())) || ((::llvm::isa<::mlir::IndexType>(elementType))) || ((::llvm::isa<::mlir::FloatType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%16, %22) 
      irdl.operands(%11)
      irdl.results(%23)
    }
    irdl.operation @gather {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ShapedType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::IntegerType>(elementType))) || ((::llvm::isa<::mlir::IndexType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %23 = irdl.all_of(%21, %22) 
      irdl.operands(%2, %3, %8, %13, %18)
      irdl.results(%23)
    }
    irdl.operation @insertelement {
      %0 = irdl.any
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      irdl.operands(%0, %3, %4)
      irdl.results(%7)
    }
    irdl.operation @insert {
      %0 = irdl.any
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %10 = irdl.all_of(%8, %9) 
      irdl.operands(%0, %5)
      irdl.results(%10)
    }
    irdl.operation @insert_strided_slice {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %14 = irdl.all_of(%12, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @load {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      irdl.operands(%2, %3)
      irdl.results(%8)
    }
    irdl.operation @mask {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.any
      %6 = irdl.any
      irdl.operands(%4, %5)
      irdl.results(%6)
    }
    irdl.operation @maskedload {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %13 = irdl.any_of(%12) 
      %14 = irdl.all_of(%11, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %20 = irdl.any_of(%19) 
      %21 = irdl.all_of(%18, %20) 
      %22 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %24 = irdl.all_of(%22, %23) 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %26 = irdl.any_of(%25) 
      %27 = irdl.all_of(%24, %26) 
      irdl.operands(%2, %3, %15, %21)
      irdl.results(%27)
    }
    irdl.operation @maskedstore {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %13 = irdl.any_of(%12) 
      %14 = irdl.all_of(%11, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %20 = irdl.any_of(%19) 
      %21 = irdl.all_of(%18, %20) 
      irdl.operands(%2, %3, %15, %21)
      irdl.results()
    }
    irdl.operation @matrix_multiply {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((elementType.isSignedInteger())) || ((::llvm::isa<::mlir::IndexType>(elementType))) || ((::llvm::isa<::mlir::FloatType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((elementType.isSignedInteger())) || ((::llvm::isa<::mlir::IndexType>(elementType))) || ((::llvm::isa<::mlir::FloatType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%16, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((elementType.isSignedInteger())) || ((::llvm::isa<::mlir::IndexType>(elementType))) || ((::llvm::isa<::mlir::FloatType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %28 = irdl.all_of(%26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %33 = irdl.any_of(%32) 
      %34 = irdl.all_of(%31, %33) 
      %35 = irdl.all_of(%28, %34) 
      irdl.operands(%11, %23)
      irdl.results(%35)
    }
    irdl.operation @multi_reduction {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.any
      %6 = irdl.any
      irdl.operands(%4, %5)
      irdl.results(%6)
    }
    irdl.operation @outerproduct {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.any
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      irdl.operands(%4, %5, %10)
      irdl.results(%15)
    }
    irdl.operation @print {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      irdl.operands(%6)
      irdl.results()
    }
    irdl.operation @reduction {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.any
      %4 = irdl.any
      irdl.operands(%2, %3)
      irdl.results(%4)
    }
    irdl.operation @reshape {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%4, %5, %6)
      irdl.results(%11)
    }
    irdl.operation @scalable.extract {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %2 = irdl.any_of(%1) 
      %3 = irdl.all_of(%0, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %8 = irdl.any_of(%7) 
      %9 = irdl.all_of(%6, %8) 
      irdl.operands(%3)
      irdl.results(%9)
    }
    irdl.operation @scalable.insert {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %8 = irdl.any_of(%7) 
      %9 = irdl.all_of(%6, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                   ::llvm::cast<VectorType>($_self).isScalable()" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %12 = irdl.any_of(%11) 
      %13 = irdl.all_of(%10, %12) 
      irdl.operands(%5, %9)
      irdl.results(%13)
    }
    irdl.operation @scan {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      irdl.operands(%4, %7)
      irdl.results(%12, %15)
    }
    irdl.operation @scatter {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::IntegerType>(elementType))) || ((::llvm::isa<::mlir::IndexType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %13 = irdl.any_of(%12) 
      %14 = irdl.all_of(%11, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %25 = irdl.any_of(%24) 
      %26 = irdl.all_of(%23, %25) 
      %27 = irdl.all_of(%20, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 1" 
      %32 = irdl.any_of(%31) 
      %33 = irdl.all_of(%30, %32) 
      irdl.operands(%2, %3, %15, %27, %33)
      irdl.results()
    }
    irdl.operation @shape_cast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @shuffle {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %10 = irdl.all_of(%8, %9) 
      irdl.operands(%2, %5)
      irdl.results(%10)
    }
    irdl.operation @splat {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %6 = irdl.all_of(%4, %5) 
      irdl.operands(%3)
      irdl.results(%6)
    }
    irdl.operation @store {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%4, %7, %8)
      irdl.results()
    }
    irdl.operation @transfer_read {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ShapedType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.any
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %12 = irdl.all_of(%10, %11) 
      irdl.operands(%2, %3, %4, %9)
      irdl.results(%12)
    }
    irdl.operation @transfer_write {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ShapedType>($_self)" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %13 = irdl.all_of(%12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      irdl.operands(%2, %5, %6, %11)
      irdl.results(%15)
    }
    irdl.operation @transpose {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @type_cast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasStaticShape()" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      irdl.operands(%4)
      irdl.results(%7)
    }
    irdl.operation @warp_execute_on_lane_0 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.any
      %2 = irdl.any
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @yield {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
  }
}
