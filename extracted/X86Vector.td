module {
  irdl.dialect @x86vector {
    irdl.operation @avx.intr.dp.ps.256 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%16, %22) 
      %24 = irdl.c_pred "$_self.isSignlessInteger(8)" 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %34 = irdl.any_of(%33) 
      %35 = irdl.all_of(%32, %34) 
      %36 = irdl.all_of(%29, %35) 
      irdl.operands(%11, %23, %24)
      irdl.results(%36)
    }
    irdl.operation @avx.intr.dot {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%16, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %28 = irdl.all_of(%26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %33 = irdl.any_of(%32) 
      %34 = irdl.all_of(%31, %33) 
      %35 = irdl.all_of(%28, %34) 
      irdl.operands(%11, %23)
      irdl.results(%35)
    }
    irdl.operation @avx512.intr.mask.compress {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isSignlessInteger(32))) || ((elementType.isF64())) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %10 = irdl.any_of(%8, %9) 
      %11 = irdl.all_of(%7, %10) 
      %12 = irdl.all_of(%4, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isSignlessInteger(32))) || ((elementType.isF64())) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
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
      %29 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %36 = irdl.any_of(%34, %35) 
      %37 = irdl.all_of(%33, %36) 
      %38 = irdl.all_of(%30, %37) 
      %39 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%12, %25, %38)
      irdl.results(%39)
    }
    irdl.operation @avx512.mask.compress {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %10 = irdl.any_of(%8, %9) 
      %11 = irdl.all_of(%7, %10) 
      %12 = irdl.all_of(%4, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isSignlessInteger(32))) || ((elementType.isF64())) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
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
      %29 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isSignlessInteger(32))) || ((elementType.isF64())) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
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
      %42 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isSignlessInteger(32))) || ((elementType.isF64())) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %49 = irdl.any_of(%47, %48) 
      %50 = irdl.all_of(%46, %49) 
      %51 = irdl.all_of(%43, %50) 
      irdl.operands(%12, %25, %38)
      irdl.results(%51)
    }
    irdl.operation @avx512.mask.rndscale {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %10 = irdl.any_of(%8, %9) 
      %11 = irdl.all_of(%7, %10) 
      %12 = irdl.all_of(%4, %11) 
      %13 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %14 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.any_of(%22, %23) 
      %25 = irdl.all_of(%21, %24) 
      %26 = irdl.all_of(%18, %25) 
      %27 = irdl.c_pred "$_self.isSignlessInteger(16)" 
      %28 = irdl.c_pred "$_self.isSignlessInteger(8)" 
      %29 = irdl.any_of(%27, %28) 
      %30 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %41 = irdl.any_of(%39, %40) 
      %42 = irdl.all_of(%38, %41) 
      %43 = irdl.all_of(%35, %42) 
      irdl.operands(%12, %13, %26, %29, %30)
      irdl.results(%43)
    }
    irdl.operation @avx512.intr.mask.rndscale.pd.512 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %13 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %17 = irdl.all_of(%15, %16) 
      %18 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %22 = irdl.any_of(%21) 
      %23 = irdl.all_of(%20, %22) 
      %24 = irdl.all_of(%17, %23) 
      %25 = irdl.c_pred "$_self.isSignlessInteger(8)" 
      %26 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %27 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%11, %12, %24, %25, %26)
      irdl.results(%27)
    }
    irdl.operation @avx512.intr.mask.rndscale.ps.512 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %13 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %17 = irdl.all_of(%15, %16) 
      %18 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %22 = irdl.any_of(%21) 
      %23 = irdl.all_of(%20, %22) 
      %24 = irdl.all_of(%17, %23) 
      %25 = irdl.c_pred "$_self.isSignlessInteger(16)" 
      %26 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %27 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%11, %12, %24, %25, %26)
      irdl.results(%27)
    }
    irdl.operation @avx512.mask.scalef {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %10 = irdl.any_of(%8, %9) 
      %11 = irdl.all_of(%7, %10) 
      %12 = irdl.all_of(%4, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
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
      %29 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %36 = irdl.any_of(%34, %35) 
      %37 = irdl.all_of(%33, %36) 
      %38 = irdl.all_of(%30, %37) 
      %39 = irdl.c_pred "$_self.isSignlessInteger(16)" 
      %40 = irdl.c_pred "$_self.isSignlessInteger(8)" 
      %41 = irdl.any_of(%39, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %53 = irdl.any_of(%51, %52) 
      %54 = irdl.all_of(%50, %53) 
      %55 = irdl.all_of(%47, %54) 
      irdl.operands(%12, %25, %38, %41, %42)
      irdl.results(%55)
    }
    irdl.operation @avx512.intr.mask.scalef.pd.512 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%16, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %28 = irdl.all_of(%26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %33 = irdl.any_of(%32) 
      %34 = irdl.all_of(%31, %33) 
      %35 = irdl.all_of(%28, %34) 
      %36 = irdl.c_pred "$_self.isSignlessInteger(8)" 
      %37 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %38 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%11, %23, %35, %36, %37)
      irdl.results(%38)
    }
    irdl.operation @avx512.intr.mask.scalef.ps.512 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%16, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %28 = irdl.all_of(%26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %33 = irdl.any_of(%32) 
      %34 = irdl.all_of(%31, %33) 
      %35 = irdl.all_of(%28, %34) 
      %36 = irdl.c_pred "$_self.isSignlessInteger(16)" 
      %37 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %38 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%11, %23, %35, %36, %37)
      irdl.results(%38)
    }
    irdl.operation @avx.intr.rsqrt.ps.256 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%11)
      irdl.results(%12)
    }
    irdl.operation @avx.rsqrt {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%16, %22) 
      irdl.operands(%11)
      irdl.results(%23)
    }
    irdl.operation @avx512.intr.vp2intersect.d.512 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%16, %22) 
      %24 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%11, %23)
      irdl.results(%24)
    }
    irdl.operation @avx512.vp2intersect {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %10 = irdl.any_of(%8, %9) 
      %11 = irdl.all_of(%7, %10) 
      %12 = irdl.all_of(%4, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
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
      %29 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
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
      %42 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %49 = irdl.any_of(%47, %48) 
      %50 = irdl.all_of(%46, %49) 
      %51 = irdl.all_of(%43, %50) 
      irdl.operands(%12, %25)
      irdl.results(%38, %51)
    }
    irdl.operation @avx512.intr.vp2intersect.q.512 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(64)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%4, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(64)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%16, %22) 
      %24 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%11, %23)
      irdl.results(%24)
    }
  }
}
