module {
  irdl.dialect @affine {
    irdl.operation @apply {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @delinearize_index {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @for {
      %0 = irdl.any
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @if {
      %0 = irdl.any
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @load {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.any
      irdl.operands(%2, %3)
      irdl.results(%4)
    }
    irdl.operation @max {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @min {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @parallel {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @prefetch {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%2, %3)
      irdl.results()
    }
    irdl.operation @store {
      %0 = irdl.any
      %1 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %3, %4)
      irdl.results()
    }
    irdl.operation @vector_load {
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
    irdl.operation @vector_store {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%4, %7, %8)
      irdl.results()
    }
    irdl.operation @yield {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
  }
}
