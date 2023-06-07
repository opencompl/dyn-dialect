module {
  irdl.dialect @arm_neon {
    irdl.operation @intr.smull {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %11 = irdl.any_of(%8, %9, %10) 
      %12 = irdl.all_of(%7, %11) 
      %13 = irdl.all_of(%4, %12) 
      %14 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %25 = irdl.any_of(%22, %23, %24) 
      %26 = irdl.all_of(%21, %25) 
      %27 = irdl.all_of(%18, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.any_of(%36, %37, %38) 
      %40 = irdl.all_of(%35, %39) 
      %41 = irdl.all_of(%32, %40) 
      irdl.operands(%13, %27)
      irdl.results(%41)
    }
    irdl.operation @"2d.sdot" {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %10 = irdl.any_of(%8, %9) 
      %11 = irdl.all_of(%7, %10) 
      %12 = irdl.all_of(%4, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %17 = irdl.all_of(%15, %16) 
      %18 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.any_of(%21, %22) 
      %24 = irdl.all_of(%20, %23) 
      %25 = irdl.all_of(%17, %24) 
      %26 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %28 = irdl.all_of(%26, %27) 
      %29 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %36 = irdl.any_of(%34, %35) 
      %37 = irdl.all_of(%33, %36) 
      %38 = irdl.all_of(%30, %37) 
      %39 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %49 = irdl.any_of(%47, %48) 
      %50 = irdl.all_of(%46, %49) 
      %51 = irdl.all_of(%43, %50) 
      irdl.operands(%12, %25, %38)
      irdl.results(%51)
    }
    irdl.operation @intr.sdot {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %10 = irdl.any_of(%8, %9) 
      %11 = irdl.all_of(%7, %10) 
      %12 = irdl.all_of(%4, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %17 = irdl.all_of(%15, %16) 
      %18 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.any_of(%21, %22) 
      %24 = irdl.all_of(%20, %23) 
      %25 = irdl.all_of(%17, %24) 
      %26 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %28 = irdl.all_of(%26, %27) 
      %29 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %36 = irdl.any_of(%34, %35) 
      %37 = irdl.all_of(%33, %36) 
      %38 = irdl.all_of(%30, %37) 
      %39 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %49 = irdl.any_of(%47, %48) 
      %50 = irdl.all_of(%46, %49) 
      %51 = irdl.all_of(%43, %50) 
      irdl.operands(%12, %25, %38)
      irdl.results(%51)
    }
  }
}
