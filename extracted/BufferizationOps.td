module {
  irdl.dialect @bufferization {
    irdl.operation @alloc_tensor {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %2 = irdl.all_of(%1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      irdl.operands(%0, %4, %5)
      irdl.results(%9)
    }
    irdl.operation @clone {
      %0 = irdl.c_pred "::llvm::isa<::mlir::BaseMemRefType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::BaseMemRefType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      irdl.operands(%3)
      irdl.results(%7)
    }
    irdl.operation @dealloc_tensor {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      irdl.operands(%3)
      irdl.results()
    }
    irdl.operation @to_memref {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::BaseMemRefType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      irdl.operands(%3)
      irdl.results(%7)
    }
    irdl.operation @to_tensor {
      %0 = irdl.c_pred "::llvm::isa<::mlir::BaseMemRefType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      irdl.operands(%3)
      irdl.results(%7)
    }
  }
}
