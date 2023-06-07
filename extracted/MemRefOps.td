module {
  irdl.dialect @memref {
    irdl.operation @assume_alignment {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      irdl.operands(%2)
      irdl.results()
    }
    irdl.operation @atomic_rmw {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<::mlir::FloatType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %7 = irdl.c_pred "$_self.isSignlessInteger()" 
      %8 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %9 = irdl.any_of(%7, %8) 
      irdl.operands(%2, %5, %6)
      irdl.results(%9)
    }
    irdl.operation @atomic_yield {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @copy {
      %0 = irdl.c_pred "::llvm::isa<::mlir::BaseMemRefType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::BaseMemRefType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      irdl.operands(%3, %7)
      irdl.results()
    }
    irdl.operation @generic_atomic_rmw {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<::mlir::FloatType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "$_self.isSignlessInteger()" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %6 = irdl.any_of(%4, %5) 
      irdl.operands(%2, %3)
      irdl.results(%6)
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
    irdl.operation @alloc {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      irdl.operands(%0, %1)
      irdl.results(%4)
    }
    irdl.operation @alloca {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      irdl.operands(%0, %1)
      irdl.results(%4)
    }
    irdl.operation @alloca_scope {
      %0 = irdl.any
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @alloca_scope.return {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @cast {
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
    irdl.operation @collapse_shape {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "isStrided(::llvm::cast<::mlir::MemRefType>($_self))" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "isStrided(::llvm::cast<::mlir::MemRefType>($_self))" 
      %9 = irdl.all_of(%7, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @dealloc {
      %0 = irdl.c_pred "::llvm::isa<::mlir::BaseMemRefType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      irdl.operands(%3)
      irdl.results()
    }
    irdl.operation @dim {
      %0 = irdl.c_pred "::llvm::isa<::mlir::UnrankedMemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank() >= 1" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.all_of(%5, %8) 
      %10 = irdl.any_of(%2, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%10, %11)
      irdl.results(%12)
    }
    irdl.operation @dma_start {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @dma_wait {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%2, %3, %4)
      irdl.results()
    }
    irdl.operation @expand_shape {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "isStrided(::llvm::cast<::mlir::MemRefType>($_self))" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "isStrided(::llvm::cast<::mlir::MemRefType>($_self))" 
      %9 = irdl.all_of(%7, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @extract_aligned_pointer_as_index {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "isStrided(::llvm::cast<::mlir::MemRefType>($_self))" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @extract_strided_metadata {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "isStrided(::llvm::cast<::mlir::MemRefType>($_self))" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "isStrided(::llvm::cast<::mlir::MemRefType>($_self))" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %11 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 0" 
      %15 = irdl.any_of(%14) 
      %16 = irdl.all_of(%13, %15) 
      %17 = irdl.all_of(%12, %16) 
      %18 = irdl.all_of(%9, %17) 
      %19 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %20 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %21 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%4)
      irdl.results(%18, %19, %20, %21)
    }
    irdl.operation @get_global {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasStaticShape()" 
      %4 = irdl.all_of(%2, %3) 
      irdl.operands()
      irdl.results(%4)
    }
    irdl.operation @global {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @memory_space_cast {
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
    irdl.operation @prefetch {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%2, %3)
      irdl.results()
    }
    irdl.operation @rank {
      %0 = irdl.c_pred "::llvm::isa<::mlir::BaseMemRefType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%3)
      irdl.results(%4)
    }
    irdl.operation @realloc {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %5 = irdl.any_of(%4) 
      %6 = irdl.all_of(%3, %5) 
      %7 = irdl.all_of(%2, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %9 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %14 = irdl.any_of(%13) 
      %15 = irdl.all_of(%12, %14) 
      %16 = irdl.all_of(%11, %15) 
      irdl.operands(%7, %8)
      irdl.results(%16)
    }
    irdl.operation @reinterpret_cast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::BaseMemRefType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      irdl.operands(%3, %4, %5, %6)
      irdl.results(%9)
    }
    irdl.operation @reshape {
      %0 = irdl.c_pred "::llvm::isa<::mlir::BaseMemRefType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %5 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<::mlir::IndexType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %8 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%6, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::BaseMemRefType>($_self)" 
      %13 = irdl.all_of(%12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      irdl.operands(%3, %11)
      irdl.results(%15)
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
    irdl.operation @transpose {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "isStrided(::llvm::cast<::mlir::MemRefType>($_self))" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "isStrided(::llvm::cast<::mlir::MemRefType>($_self))" 
      %9 = irdl.all_of(%7, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @view {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %5 = irdl.any_of(%4) 
      %6 = irdl.all_of(%3, %5) 
      %7 = irdl.all_of(%2, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %9 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %10 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %11 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %12 = irdl.all_of(%10, %11) 
      irdl.operands(%7, %8, %9)
      irdl.results(%12)
    }
    irdl.operation @subview {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      irdl.operands(%2, %3, %4, %5)
      irdl.results(%8)
    }
    irdl.operation @tensor_store {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::BaseMemRefType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      irdl.operands(%3, %7)
      irdl.results()
    }
  }
}
