module {
  irdl.dialect @amx {
    irdl.operation @tdpbf16ps {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4, %5)
      irdl.results(%6)
    }
    irdl.operation @tdpbssd {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4, %5)
      irdl.results(%6)
    }
    irdl.operation @tdpbsud {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4, %5)
      irdl.results(%6)
    }
    irdl.operation @tdpbusd {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4, %5)
      irdl.results(%6)
    }
    irdl.operation @tdpbuud {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4, %5)
      irdl.results(%6)
    }
    irdl.operation @tileloadd64 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3)
      irdl.results(%4)
    }
    irdl.operation @tilestored64 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2, %3, %4)
      irdl.results()
    }
    irdl.operation @tilezero {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @tile_load {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isBF16())) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(8))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 2" 
      %13 = irdl.any_of(%12) 
      %14 = irdl.all_of(%11, %13) 
      %15 = irdl.all_of(%8, %14) 
      irdl.operands(%2, %3)
      irdl.results(%15)
    }
    irdl.operation @tile_mulf {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isBF16())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 2" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isBF16())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 2" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%16, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isBF16())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %28 = irdl.all_of(%26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 2" 
      %33 = irdl.any_of(%32) 
      %34 = irdl.all_of(%31, %33) 
      %35 = irdl.all_of(%28, %34) 
      %36 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isBF16())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %40 = irdl.all_of(%38, %39) 
      %41 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 2" 
      %45 = irdl.any_of(%44) 
      %46 = irdl.all_of(%43, %45) 
      %47 = irdl.all_of(%40, %46) 
      irdl.operands(%11, %23, %35)
      irdl.results(%47)
    }
    irdl.operation @tile_muli {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(8))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 2" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(8))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 2" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%16, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(8))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %28 = irdl.all_of(%26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 2" 
      %33 = irdl.any_of(%32) 
      %34 = irdl.all_of(%31, %33) 
      %35 = irdl.all_of(%28, %34) 
      %36 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(8))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %40 = irdl.all_of(%38, %39) 
      %41 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 2" 
      %45 = irdl.any_of(%44) 
      %46 = irdl.all_of(%43, %45) 
      %47 = irdl.all_of(%40, %46) 
      irdl.operands(%11, %23, %35)
      irdl.results(%47)
    }
    irdl.operation @tile_store {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isBF16())) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(8))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 2" 
      %13 = irdl.any_of(%12) 
      %14 = irdl.all_of(%11, %13) 
      %15 = irdl.all_of(%8, %14) 
      irdl.operands(%2, %3, %15)
      irdl.results()
    }
    irdl.operation @tile_zero {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isBF16())) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(8))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank()\0A                           == 2" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      irdl.operands()
      irdl.results(%11)
    }
  }
}
