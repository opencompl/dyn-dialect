module {
  irdl.dialect @spirv {
    irdl.operation @CompositeConstruct {
      %0 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      %7 = irdl.c_pred "$_self.isF16()" 
      %8 = irdl.c_pred "$_self.isF32()" 
      %9 = irdl.c_pred "$_self.isF64()" 
      %10 = irdl.any_of(%7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %24 = irdl.any_of(%19, %20, %21, %22, %23) 
      %25 = irdl.all_of(%18, %24) 
      %26 = irdl.all_of(%15, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %35 = irdl.any_of(%0, %1, %6, %10, %26, %27, %28, %29, %30, %31, %32, %33, %34) 
      %36 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %40 = irdl.all_of(%38, %39) 
      %41 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %49 = irdl.any_of(%44, %45, %46, %47, %48) 
      %50 = irdl.all_of(%43, %49) 
      %51 = irdl.all_of(%40, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %53 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %54 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %55 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %56 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %57 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %58 = irdl.any_of(%51, %52, %53, %54, %55, %56, %57) 
      irdl.operands(%35)
      irdl.results(%58)
    }
    irdl.operation @CompositeExtract {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %13 = irdl.any_of(%8, %9, %10, %11, %12) 
      %14 = irdl.all_of(%7, %13) 
      %15 = irdl.all_of(%4, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %17 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %18 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %19 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %22 = irdl.any_of(%15, %16, %17, %18, %19, %20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %24 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "$_self.isF16()" 
      %31 = irdl.c_pred "$_self.isF32()" 
      %32 = irdl.c_pred "$_self.isF64()" 
      %33 = irdl.any_of(%30, %31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %47 = irdl.any_of(%42, %43, %44, %45, %46) 
      %48 = irdl.all_of(%41, %47) 
      %49 = irdl.all_of(%38, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %51 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %52 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %53 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %54 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %55 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %56 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %57 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %58 = irdl.any_of(%23, %24, %29, %33, %49, %50, %51, %52, %53, %54, %55, %56, %57) 
      irdl.operands(%22)
      irdl.results(%58)
    }
    irdl.operation @CompositeInsert {
      %0 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      %7 = irdl.c_pred "$_self.isF16()" 
      %8 = irdl.c_pred "$_self.isF32()" 
      %9 = irdl.c_pred "$_self.isF64()" 
      %10 = irdl.any_of(%7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %24 = irdl.any_of(%19, %20, %21, %22, %23) 
      %25 = irdl.all_of(%18, %24) 
      %26 = irdl.all_of(%15, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %35 = irdl.any_of(%0, %1, %6, %10, %26, %27, %28, %29, %30, %31, %32, %33, %34) 
      %36 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %40 = irdl.all_of(%38, %39) 
      %41 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %49 = irdl.any_of(%44, %45, %46, %47, %48) 
      %50 = irdl.all_of(%43, %49) 
      %51 = irdl.all_of(%40, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %53 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %54 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %55 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %56 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %57 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %58 = irdl.any_of(%51, %52, %53, %54, %55, %56, %57) 
      %59 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %61 = irdl.all_of(%59, %60) 
      %62 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %63 = irdl.all_of(%61, %62) 
      %64 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %66 = irdl.all_of(%64, %65) 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %68 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %69 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %70 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %71 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %72 = irdl.any_of(%67, %68, %69, %70, %71) 
      %73 = irdl.all_of(%66, %72) 
      %74 = irdl.all_of(%63, %73) 
      %75 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %76 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %77 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %78 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %79 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %80 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %81 = irdl.any_of(%74, %75, %76, %77, %78, %79, %80) 
      irdl.operands(%35, %58)
      irdl.results(%81)
    }
    irdl.operation @VectorExtractDynamic {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %13 = irdl.any_of(%8, %9, %10, %11, %12) 
      %14 = irdl.all_of(%7, %13) 
      %15 = irdl.all_of(%4, %14) 
      %16 = irdl.c_pred "$_self.isInteger(8)" 
      %17 = irdl.c_pred "$_self.isInteger(16)" 
      %18 = irdl.c_pred "$_self.isInteger(32)" 
      %19 = irdl.c_pred "$_self.isInteger(64)" 
      %20 = irdl.any_of(%16, %17, %18, %19) 
      %21 = irdl.c_pred "$_self.isInteger(8)" 
      %22 = irdl.c_pred "$_self.isInteger(16)" 
      %23 = irdl.c_pred "$_self.isInteger(32)" 
      %24 = irdl.c_pred "$_self.isInteger(64)" 
      %25 = irdl.any_of(%21, %22, %23, %24) 
      %26 = irdl.c_pred "$_self.isF16()" 
      %27 = irdl.c_pred "$_self.isF32()" 
      %28 = irdl.c_pred "$_self.isF64()" 
      %29 = irdl.any_of(%26, %27, %28) 
      %30 = irdl.any_of(%25, %29) 
      %31 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %32 = irdl.any_of(%30, %31) 
      irdl.operands(%15, %20)
      irdl.results(%32)
    }
    irdl.operation @VectorInsertDynamic {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %13 = irdl.any_of(%8, %9, %10, %11, %12) 
      %14 = irdl.all_of(%7, %13) 
      %15 = irdl.all_of(%4, %14) 
      %16 = irdl.c_pred "$_self.isInteger(8)" 
      %17 = irdl.c_pred "$_self.isInteger(16)" 
      %18 = irdl.c_pred "$_self.isInteger(32)" 
      %19 = irdl.c_pred "$_self.isInteger(64)" 
      %20 = irdl.any_of(%16, %17, %18, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.any_of(%20, %24) 
      %26 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %27 = irdl.any_of(%25, %26) 
      %28 = irdl.c_pred "$_self.isInteger(8)" 
      %29 = irdl.c_pred "$_self.isInteger(16)" 
      %30 = irdl.c_pred "$_self.isInteger(32)" 
      %31 = irdl.c_pred "$_self.isInteger(64)" 
      %32 = irdl.any_of(%28, %29, %30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %40 = irdl.all_of(%38, %39) 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %46 = irdl.any_of(%41, %42, %43, %44, %45) 
      %47 = irdl.all_of(%40, %46) 
      %48 = irdl.all_of(%37, %47) 
      irdl.operands(%15, %27, %32)
      irdl.results(%48)
    }
    irdl.operation @VectorShuffle {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %13 = irdl.any_of(%8, %9, %10, %11, %12) 
      %14 = irdl.all_of(%7, %13) 
      %15 = irdl.all_of(%4, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %29 = irdl.any_of(%24, %25, %26, %27, %28) 
      %30 = irdl.all_of(%23, %29) 
      %31 = irdl.all_of(%20, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %45 = irdl.any_of(%40, %41, %42, %43, %44) 
      %46 = irdl.all_of(%39, %45) 
      %47 = irdl.all_of(%36, %46) 
      irdl.operands(%15, %31)
      irdl.results(%47)
    }
  }
}
