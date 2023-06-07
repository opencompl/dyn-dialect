module {
  irdl.dialect @spirv {
    irdl.operation @Bitcast {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "$_self.isF16()" 
      %6 = irdl.c_pred "$_self.isF32()" 
      %7 = irdl.c_pred "$_self.isF64()" 
      %8 = irdl.any_of(%5, %6, %7) 
      %9 = irdl.any_of(%4, %8) 
      %10 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %11 = irdl.any_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %25 = irdl.any_of(%20, %21, %22, %23, %24) 
      %26 = irdl.all_of(%19, %25) 
      %27 = irdl.all_of(%16, %26) 
      %28 = irdl.any_of(%11, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %30 = irdl.any_of(%28, %29) 
      %31 = irdl.c_pred "$_self.isInteger(8)" 
      %32 = irdl.c_pred "$_self.isInteger(16)" 
      %33 = irdl.c_pred "$_self.isInteger(32)" 
      %34 = irdl.c_pred "$_self.isInteger(64)" 
      %35 = irdl.any_of(%31, %32, %33, %34) 
      %36 = irdl.c_pred "$_self.isF16()" 
      %37 = irdl.c_pred "$_self.isF32()" 
      %38 = irdl.c_pred "$_self.isF64()" 
      %39 = irdl.any_of(%36, %37, %38) 
      %40 = irdl.any_of(%35, %39) 
      %41 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %42 = irdl.any_of(%40, %41) 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %61 = irdl.any_of(%59, %60) 
      irdl.operands(%30)
      irdl.results(%61)
    }
    irdl.operation @ConvertFToS {
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
      %24 = irdl.c_pred "$_self.isInteger(8)" 
      %25 = irdl.c_pred "$_self.isInteger(16)" 
      %26 = irdl.c_pred "$_self.isInteger(32)" 
      %27 = irdl.c_pred "$_self.isInteger(64)" 
      %28 = irdl.any_of(%24, %25, %26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %42 = irdl.any_of(%37, %38, %39, %40, %41) 
      %43 = irdl.all_of(%36, %42) 
      %44 = irdl.all_of(%33, %43) 
      %45 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %46 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.any_of(%28, %44, %47) 
      irdl.operands(%23)
      irdl.results(%48)
    }
    irdl.operation @ConvertFToU {
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
      %24 = irdl.c_pred "$_self.isInteger(8)" 
      %25 = irdl.c_pred "$_self.isInteger(16)" 
      %26 = irdl.c_pred "$_self.isInteger(32)" 
      %27 = irdl.c_pred "$_self.isInteger(64)" 
      %28 = irdl.any_of(%24, %25, %26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %42 = irdl.any_of(%37, %38, %39, %40, %41) 
      %43 = irdl.all_of(%36, %42) 
      %44 = irdl.all_of(%33, %43) 
      %45 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %46 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.any_of(%28, %44, %47) 
      irdl.operands(%23)
      irdl.results(%48)
    }
    irdl.operation @ConvertPtrToU {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      irdl.operands(%0)
      irdl.results(%5)
    }
    irdl.operation @ConvertSToF {
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
      %25 = irdl.c_pred "$_self.isF16()" 
      %26 = irdl.c_pred "$_self.isF32()" 
      %27 = irdl.c_pred "$_self.isF64()" 
      %28 = irdl.any_of(%25, %26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %42 = irdl.any_of(%37, %38, %39, %40, %41) 
      %43 = irdl.all_of(%36, %42) 
      %44 = irdl.all_of(%33, %43) 
      %45 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %46 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.any_of(%28, %44, %47) 
      irdl.operands(%24)
      irdl.results(%48)
    }
    irdl.operation @ConvertUToF {
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
      %25 = irdl.c_pred "$_self.isF16()" 
      %26 = irdl.c_pred "$_self.isF32()" 
      %27 = irdl.c_pred "$_self.isF64()" 
      %28 = irdl.any_of(%25, %26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %42 = irdl.any_of(%37, %38, %39, %40, %41) 
      %43 = irdl.all_of(%36, %42) 
      %44 = irdl.all_of(%33, %43) 
      %45 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %46 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.any_of(%28, %44, %47) 
      irdl.operands(%24)
      irdl.results(%48)
    }
    irdl.operation @ConvertUToPtr {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @FConvert {
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
    irdl.operation @GenericCastToPtrExplicit {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @GenericCastToPtr {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @PtrCastToGeneric {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @SConvert {
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
    irdl.operation @UConvert {
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
  }
}
