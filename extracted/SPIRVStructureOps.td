module {
  irdl.dialect @spirv {
    irdl.operation @mlir.addressof {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @Constant {
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
      irdl.operands()
      irdl.results(%35)
    }
    irdl.operation @EntryPoint {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @ExecutionMode {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @func {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @GlobalVariable {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @module {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @mlir.referenceof {
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
      irdl.operands()
      irdl.results(%35)
    }
    irdl.operation @SpecConstantComposite {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @SpecConstant {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @SpecConstantOperation {
      %0 = irdl.any
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @mlir.yield {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
  }
}
