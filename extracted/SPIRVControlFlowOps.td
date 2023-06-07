module {
  irdl.dialect @spirv {
    irdl.operation @BranchConditional {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %3 = irdl.c_pred "$_self.isInteger(8)" 
      %4 = irdl.c_pred "$_self.isInteger(16)" 
      %5 = irdl.c_pred "$_self.isInteger(32)" 
      %6 = irdl.c_pred "$_self.isInteger(64)" 
      %7 = irdl.any_of(%3, %4, %5, %6) 
      %8 = irdl.c_pred "$_self.isF16()" 
      %9 = irdl.c_pred "$_self.isF32()" 
      %10 = irdl.c_pred "$_self.isF64()" 
      %11 = irdl.any_of(%8, %9, %10) 
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
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %35 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %36 = irdl.any_of(%1, %2, %7, %11, %27, %28, %29, %30, %31, %32, %33, %34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %38 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %39 = irdl.c_pred "$_self.isInteger(8)" 
      %40 = irdl.c_pred "$_self.isInteger(16)" 
      %41 = irdl.c_pred "$_self.isInteger(32)" 
      %42 = irdl.c_pred "$_self.isInteger(64)" 
      %43 = irdl.any_of(%39, %40, %41, %42) 
      %44 = irdl.c_pred "$_self.isF16()" 
      %45 = irdl.c_pred "$_self.isF32()" 
      %46 = irdl.c_pred "$_self.isF64()" 
      %47 = irdl.any_of(%44, %45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %55 = irdl.all_of(%53, %54) 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %61 = irdl.any_of(%56, %57, %58, %59, %60) 
      %62 = irdl.all_of(%55, %61) 
      %63 = irdl.all_of(%52, %62) 
      %64 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %65 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %66 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %67 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %70 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %72 = irdl.any_of(%37, %38, %43, %47, %63, %64, %65, %66, %67, %68, %69, %70, %71) 
      irdl.operands(%0, %36, %72)
      irdl.results()
    }
    irdl.operation @Branch {
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
      irdl.operands(%35)
      irdl.results()
    }
    irdl.operation @FunctionCall {
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
      %36 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %37 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %38 = irdl.c_pred "$_self.isInteger(8)" 
      %39 = irdl.c_pred "$_self.isInteger(16)" 
      %40 = irdl.c_pred "$_self.isInteger(32)" 
      %41 = irdl.c_pred "$_self.isInteger(64)" 
      %42 = irdl.any_of(%38, %39, %40, %41) 
      %43 = irdl.c_pred "$_self.isF16()" 
      %44 = irdl.c_pred "$_self.isF32()" 
      %45 = irdl.c_pred "$_self.isF64()" 
      %46 = irdl.any_of(%43, %44, %45) 
      %47 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %60 = irdl.any_of(%55, %56, %57, %58, %59) 
      %61 = irdl.all_of(%54, %60) 
      %62 = irdl.all_of(%51, %61) 
      %63 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %64 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %65 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %66 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %67 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %69 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %70 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %71 = irdl.any_of(%36, %37, %42, %46, %62, %63, %64, %65, %66, %67, %68, %69, %70) 
      irdl.operands(%35)
      irdl.results(%71)
    }
    irdl.operation @mlir.loop {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @mlir.merge {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @Return {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @ReturnValue {
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
      irdl.operands(%35)
      irdl.results()
    }
    irdl.operation @mlir.selection {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @Unreachable {
      irdl.operands()
      irdl.results()
    }
  }
}
