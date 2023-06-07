module {
  irdl.dialect @nvgpu {
    irdl.operation @device_async_copy {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %5 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %8 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %9 = irdl.c_pred "::llvm::isa<::mlir::nvgpu::DeviceAsyncTokenType>($_self)" 
      irdl.operands(%2, %3, %6, %7, %8)
      irdl.results(%9)
    }
    irdl.operation @device_async_create_group {
      %0 = irdl.c_pred "::llvm::isa<::mlir::nvgpu::DeviceAsyncTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::nvgpu::DeviceAsyncTokenType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @device_async_wait {
      %0 = irdl.c_pred "::llvm::isa<::mlir::nvgpu::DeviceAsyncTokenType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @ldmatrix {
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
    irdl.operation @mma.sp.sync {
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
      %15 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                  !::llvm::cast<VectorType>($_self).isScalable()" 
      %16 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(16)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %17 = irdl.all_of(%15, %16) 
      %18 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self) &&\0A                                  !::llvm::cast<VectorType>($_self).isScalable()" 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %20 = irdl.any_of(%19) 
      %21 = irdl.all_of(%18, %20) 
      %22 = irdl.all_of(%17, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      irdl.operands(%4, %9, %14, %22)
      irdl.results(%27)
    }
    irdl.operation @mma.sync {
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
      %15 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %17 = irdl.all_of(%15, %16) 
      %18 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %19 = irdl.all_of(%17, %18) 
      irdl.operands(%4, %9, %14)
      irdl.results(%19)
    }
  }
}
