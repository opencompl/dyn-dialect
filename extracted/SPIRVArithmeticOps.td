module {
  irdl.dialect @spirv {
    irdl.operation @FAdd {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      %48 = irdl.c_pred "$_self.isF16()" 
      %49 = irdl.c_pred "$_self.isF32()" 
      %50 = irdl.c_pred "$_self.isF64()" 
      %51 = irdl.any_of(%48, %49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %65 = irdl.any_of(%60, %61, %62, %63, %64) 
      %66 = irdl.all_of(%59, %65) 
      %67 = irdl.all_of(%56, %66) 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %70 = irdl.all_of(%68, %69) 
      %71 = irdl.any_of(%51, %67, %70) 
      irdl.operands(%23, %47)
      irdl.results(%71)
    }
    irdl.operation @FDiv {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      %48 = irdl.c_pred "$_self.isF16()" 
      %49 = irdl.c_pred "$_self.isF32()" 
      %50 = irdl.c_pred "$_self.isF64()" 
      %51 = irdl.any_of(%48, %49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %65 = irdl.any_of(%60, %61, %62, %63, %64) 
      %66 = irdl.all_of(%59, %65) 
      %67 = irdl.all_of(%56, %66) 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %70 = irdl.all_of(%68, %69) 
      %71 = irdl.any_of(%51, %67, %70) 
      irdl.operands(%23, %47)
      irdl.results(%71)
    }
    irdl.operation @FMod {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      %48 = irdl.c_pred "$_self.isF16()" 
      %49 = irdl.c_pred "$_self.isF32()" 
      %50 = irdl.c_pred "$_self.isF64()" 
      %51 = irdl.any_of(%48, %49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %65 = irdl.any_of(%60, %61, %62, %63, %64) 
      %66 = irdl.all_of(%59, %65) 
      %67 = irdl.all_of(%56, %66) 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %70 = irdl.all_of(%68, %69) 
      %71 = irdl.any_of(%51, %67, %70) 
      irdl.operands(%23, %47)
      irdl.results(%71)
    }
    irdl.operation @FMul {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      %48 = irdl.c_pred "$_self.isF16()" 
      %49 = irdl.c_pred "$_self.isF32()" 
      %50 = irdl.c_pred "$_self.isF64()" 
      %51 = irdl.any_of(%48, %49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %65 = irdl.any_of(%60, %61, %62, %63, %64) 
      %66 = irdl.all_of(%59, %65) 
      %67 = irdl.all_of(%56, %66) 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %70 = irdl.all_of(%68, %69) 
      %71 = irdl.any_of(%51, %67, %70) 
      irdl.operands(%23, %47)
      irdl.results(%71)
    }
    irdl.operation @FNegate {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      irdl.operands(%23)
      irdl.results(%47)
    }
    irdl.operation @FRem {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      %48 = irdl.c_pred "$_self.isF16()" 
      %49 = irdl.c_pred "$_self.isF32()" 
      %50 = irdl.c_pred "$_self.isF64()" 
      %51 = irdl.any_of(%48, %49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %65 = irdl.any_of(%60, %61, %62, %63, %64) 
      %66 = irdl.all_of(%59, %65) 
      %67 = irdl.all_of(%56, %66) 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %70 = irdl.all_of(%68, %69) 
      %71 = irdl.any_of(%51, %67, %70) 
      irdl.operands(%23, %47)
      irdl.results(%71)
    }
    irdl.operation @FSub {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      %48 = irdl.c_pred "$_self.isF16()" 
      %49 = irdl.c_pred "$_self.isF32()" 
      %50 = irdl.c_pred "$_self.isF64()" 
      %51 = irdl.any_of(%48, %49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %65 = irdl.any_of(%60, %61, %62, %63, %64) 
      %66 = irdl.all_of(%59, %65) 
      %67 = irdl.all_of(%56, %66) 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %70 = irdl.all_of(%68, %69) 
      %71 = irdl.any_of(%51, %67, %70) 
      irdl.operands(%23, %47)
      irdl.results(%71)
    }
    irdl.operation @IAddCarry {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.any_of(%4, %20) 
      %22 = irdl.c_pred "$_self.isInteger(8)" 
      %23 = irdl.c_pred "$_self.isInteger(16)" 
      %24 = irdl.c_pred "$_self.isInteger(32)" 
      %25 = irdl.c_pred "$_self.isInteger(64)" 
      %26 = irdl.any_of(%22, %23, %24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %40 = irdl.any_of(%35, %36, %37, %38, %39) 
      %41 = irdl.all_of(%34, %40) 
      %42 = irdl.all_of(%31, %41) 
      %43 = irdl.any_of(%26, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      irdl.operands(%21, %43)
      irdl.results(%44)
    }
    irdl.operation @IAdd {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
    irdl.operation @IMul {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
    irdl.operation @ISubBorrow {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.any_of(%4, %20) 
      %22 = irdl.c_pred "$_self.isInteger(8)" 
      %23 = irdl.c_pred "$_self.isInteger(16)" 
      %24 = irdl.c_pred "$_self.isInteger(32)" 
      %25 = irdl.c_pred "$_self.isInteger(64)" 
      %26 = irdl.any_of(%22, %23, %24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %40 = irdl.any_of(%35, %36, %37, %38, %39) 
      %41 = irdl.all_of(%34, %40) 
      %42 = irdl.all_of(%31, %41) 
      %43 = irdl.any_of(%26, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      irdl.operands(%21, %43)
      irdl.results(%44)
    }
    irdl.operation @ISub {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
    irdl.operation @SDiv {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
    irdl.operation @SMod {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
    irdl.operation @SMulExtended {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.any_of(%4, %20) 
      %22 = irdl.c_pred "$_self.isInteger(8)" 
      %23 = irdl.c_pred "$_self.isInteger(16)" 
      %24 = irdl.c_pred "$_self.isInteger(32)" 
      %25 = irdl.c_pred "$_self.isInteger(64)" 
      %26 = irdl.any_of(%22, %23, %24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %40 = irdl.any_of(%35, %36, %37, %38, %39) 
      %41 = irdl.all_of(%34, %40) 
      %42 = irdl.all_of(%31, %41) 
      %43 = irdl.any_of(%26, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      irdl.operands(%21, %43)
      irdl.results(%44)
    }
    irdl.operation @SNegate {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      irdl.operands(%24)
      irdl.results(%49)
    }
    irdl.operation @SRem {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
    irdl.operation @UDiv {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
    irdl.operation @UMod {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
    irdl.operation @UMulExtended {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.any_of(%4, %20) 
      %22 = irdl.c_pred "$_self.isInteger(8)" 
      %23 = irdl.c_pred "$_self.isInteger(16)" 
      %24 = irdl.c_pred "$_self.isInteger(32)" 
      %25 = irdl.c_pred "$_self.isInteger(64)" 
      %26 = irdl.any_of(%22, %23, %24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %40 = irdl.any_of(%35, %36, %37, %38, %39) 
      %41 = irdl.all_of(%34, %40) 
      %42 = irdl.all_of(%31, %41) 
      %43 = irdl.any_of(%26, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      irdl.operands(%21, %43)
      irdl.results(%44)
    }
    irdl.operation @VectorTimesScalar {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %11 = irdl.any_of(%8, %9, %10) 
      %12 = irdl.all_of(%7, %11) 
      %13 = irdl.all_of(%4, %12) 
      %14 = irdl.c_pred "$_self.isF16()" 
      %15 = irdl.c_pred "$_self.isF32()" 
      %16 = irdl.c_pred "$_self.isF64()" 
      %17 = irdl.any_of(%14, %15, %16) 
      %18 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %29 = irdl.any_of(%26, %27, %28) 
      %30 = irdl.all_of(%25, %29) 
      %31 = irdl.all_of(%22, %30) 
      irdl.operands(%13, %17)
      irdl.results(%31)
    }
  }
}
