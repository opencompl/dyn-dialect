module {
  irdl.dialect @tensor {
    irdl.operation @bitcast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
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
    irdl.operation @cast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
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
    irdl.operation @collapse_shape {
      %0 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      irdl.operands(%3)
      irdl.results(%7)
    }
    irdl.operation @dim {
      %0 = irdl.c_pred "::llvm::isa<::mlir::UnrankedTensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank() >= 1" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.all_of(%4, %7) 
      %9 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.any_of(%3, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %13 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%11, %12)
      irdl.results(%13)
    }
    irdl.operation @empty {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.all_of(%1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      irdl.operands(%0)
      irdl.results(%4)
    }
    irdl.operation @expand_shape {
      %0 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      irdl.operands(%3)
      irdl.results(%7)
    }
    irdl.operation @extract {
      %0 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.any
      irdl.operands(%3, %4)
      irdl.results(%5)
    }
    irdl.operation @extract_slice {
      %0 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %8 = irdl.all_of(%7) 
      %9 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %10 = irdl.all_of(%8, %9) 
      irdl.operands(%3, %4, %5, %6)
      irdl.results(%10)
    }
    irdl.operation @from_elements {
      %0 = irdl.any
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasStaticShape()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%0)
      irdl.results(%5)
    }
    irdl.operation @gather {
      %0 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %9 = irdl.all_of(%8) 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%3, %7)
      irdl.results(%11)
    }
    irdl.operation @generate {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.all_of(%1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      irdl.operands(%0)
      irdl.results(%4)
    }
    irdl.operation @insert {
      %0 = irdl.any
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.all_of(%1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      irdl.operands(%0, %4, %5)
      irdl.results(%9)
    }
    irdl.operation @insert_slice {
      %0 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %9 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %11 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %12 = irdl.all_of(%11) 
      %13 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %14 = irdl.all_of(%12, %13) 
      irdl.operands(%3, %7, %8, %9, %10)
      irdl.results(%14)
    }
    irdl.operation @pack {
      %0 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.any
      %9 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %10 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %11 = irdl.all_of(%10) 
      %12 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %13 = irdl.all_of(%11, %12) 
      irdl.operands(%3, %7, %8, %9)
      irdl.results(%13)
    }
    irdl.operation @pad {
      %0 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      irdl.operands(%3, %4, %5)
      irdl.results(%9)
    }
    irdl.operation @parallel_insert_slice {
      %0 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %9 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%3, %7, %8, %9, %10)
      irdl.results()
    }
    irdl.operation @rank {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%3)
      irdl.results(%4)
    }
    irdl.operation @reshape {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %7 = irdl.any_of(%6) 
      %8 = irdl.all_of(%5, %7) 
      %9 = irdl.all_of(%4, %8) 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<::mlir::IndexType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %13 = irdl.all_of(%12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      irdl.operands(%3, %11)
      irdl.results(%15)
    }
    irdl.operation @scatter {
      %0 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %9 = irdl.all_of(%8) 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %13 = irdl.all_of(%12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      irdl.operands(%3, %7, %11)
      irdl.results(%15)
    }
    irdl.operation @splat {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasStaticShape()" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      irdl.operands(%3)
      irdl.results(%8)
    }
    irdl.operation @unpack {
      %0 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %9 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %10 = irdl.all_of(%9) 
      %11 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %12 = irdl.all_of(%10, %11) 
      irdl.operands(%3, %7, %8)
      irdl.results(%12)
    }
    irdl.operation @yield {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
  }
}
