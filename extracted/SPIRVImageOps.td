module {
  irdl.dialect @spirv {
    irdl.operation @ImageDrefGather {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %1 = irdl.c_pred "$_self.isF16()" 
      %2 = irdl.c_pred "$_self.isF32()" 
      %3 = irdl.c_pred "$_self.isF64()" 
      %4 = irdl.any_of(%1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
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
      %22 = irdl.c_pred "$_self.isF16()" 
      %23 = irdl.c_pred "$_self.isF32()" 
      %24 = irdl.c_pred "$_self.isF64()" 
      %25 = irdl.any_of(%22, %23, %24) 
      %26 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %27 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %28 = irdl.c_pred "$_self.isInteger(8)" 
      %29 = irdl.c_pred "$_self.isInteger(16)" 
      %30 = irdl.c_pred "$_self.isInteger(32)" 
      %31 = irdl.c_pred "$_self.isInteger(64)" 
      %32 = irdl.any_of(%28, %29, %30, %31) 
      %33 = irdl.c_pred "$_self.isF16()" 
      %34 = irdl.c_pred "$_self.isF32()" 
      %35 = irdl.c_pred "$_self.isF64()" 
      %36 = irdl.any_of(%33, %34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %44 = irdl.all_of(%42, %43) 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %50 = irdl.any_of(%45, %46, %47, %48, %49) 
      %51 = irdl.all_of(%44, %50) 
      %52 = irdl.all_of(%41, %51) 
      %53 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %54 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %55 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %56 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %57 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %58 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %59 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %60 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %61 = irdl.any_of(%26, %27, %32, %36, %52, %53, %54, %55, %56, %57, %58, %59, %60) 
      %62 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %64 = irdl.all_of(%62, %63) 
      %65 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %66 = irdl.all_of(%64, %65) 
      %67 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %68 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %69 = irdl.all_of(%67, %68) 
      %70 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %71 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %72 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %73 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %74 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %75 = irdl.any_of(%70, %71, %72, %73, %74) 
      %76 = irdl.all_of(%69, %75) 
      %77 = irdl.all_of(%66, %76) 
      irdl.operands(%0, %21, %25, %61)
      irdl.results(%77)
    }
    irdl.operation @Image {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::ImageType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @ImageQuerySize {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::ImageType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %19 = irdl.any_of(%14, %15, %16, %17, %18) 
      %20 = irdl.all_of(%13, %19) 
      %21 = irdl.all_of(%10, %20) 
      %22 = irdl.any_of(%5, %21) 
      irdl.operands(%0)
      irdl.results(%22)
    }
  }
}
