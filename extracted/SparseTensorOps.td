module {
  irdl.dialect @sparse_tensor {
    irdl.operation @binary {
      %0 = irdl.any
      %1 = irdl.any
      %2 = irdl.any
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @compress {
      %0 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "isStrided(::llvm::cast<::mlir::MemRefType>($_self))" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %10 = irdl.any_of(%9) 
      %11 = irdl.all_of(%8, %10) 
      %12 = irdl.all_of(%7, %11) 
      %13 = irdl.all_of(%4, %12) 
      %14 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %19 = irdl.any_of(%18) 
      %20 = irdl.all_of(%17, %19) 
      %21 = irdl.all_of(%16, %20) 
      %22 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %23 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %24 = irdl.all_of(%22, %23) 
      %25 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %27 = irdl.any_of(%26) 
      %28 = irdl.all_of(%25, %27) 
      %29 = irdl.all_of(%24, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %32 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %37 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %38 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %41 = irdl.all_of(%39, %40) 
      irdl.operands(%13, %21, %29, %30, %35, %36)
      irdl.results(%41)
    }
    irdl.operation @concatenate {
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
    irdl.operation @convert {
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
    irdl.operation @expand {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
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
      %14 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %15 = irdl.any_of(%14) 
      %16 = irdl.all_of(%13, %15) 
      %17 = irdl.all_of(%12, %16) 
      %18 = irdl.all_of(%9, %17) 
      %19 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %20 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %24 = irdl.any_of(%23) 
      %25 = irdl.all_of(%22, %24) 
      %26 = irdl.all_of(%21, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %32 = irdl.any_of(%31) 
      %33 = irdl.all_of(%30, %32) 
      %34 = irdl.all_of(%29, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%4)
      irdl.results(%18, %26, %34, %35)
    }
    irdl.operation @foreach {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any
      %5 = irdl.any
      irdl.operands(%3, %4)
      irdl.results(%5)
    }
    irdl.operation @storage_specifier.get {
      %0 = irdl.c_pred "::llvm::isa<::mlir::sparse_tensor::StorageSpecifierType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @insert {
      %0 = irdl.any
      %1 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %2 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %8 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%0, %5, %6)
      irdl.results(%11)
    }
    irdl.operation @load {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %6 = irdl.all_of(%5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      irdl.operands(%4)
      irdl.results(%8)
    }
    irdl.operation @new {
      %0 = irdl.any
      %1 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %2 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%0)
      irdl.results(%5)
    }
    irdl.operation @number_of_entries {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @out {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.any
      irdl.operands(%4, %5)
      irdl.results()
    }
    irdl.operation @pack {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.all_of(%0) 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %9 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %12 = irdl.all_of(%10, %11) 
      irdl.operands(%3, %7)
      irdl.results(%12)
    }
    irdl.operation @push_back {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %6 = irdl.any_of(%5) 
      %7 = irdl.all_of(%4, %6) 
      %8 = irdl.all_of(%3, %7) 
      %9 = irdl.any
      %10 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %11 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %12 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %16 = irdl.any_of(%15) 
      %17 = irdl.all_of(%14, %16) 
      %18 = irdl.all_of(%13, %17) 
      %19 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %8, %9, %10)
      irdl.results(%18, %19)
    }
    irdl.operation @reduce {
      %0 = irdl.any
      %1 = irdl.any
      %2 = irdl.any
      %3 = irdl.any
      irdl.operands(%0, %1, %2)
      irdl.results(%3)
    }
    irdl.operation @select {
      %0 = irdl.any
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @storage_specifier.set {
      %0 = irdl.c_pred "::llvm::isa<::mlir::sparse_tensor::StorageSpecifierType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::sparse_tensor::StorageSpecifierType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @sort_coo {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::IntegerType>(elementType))) || ((::llvm::isa<::mlir::IndexType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %6 = irdl.any_of(%5) 
      %7 = irdl.all_of(%4, %6) 
      %8 = irdl.all_of(%3, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %14 = irdl.any_of(%13) 
      %15 = irdl.all_of(%12, %14) 
      %16 = irdl.all_of(%11, %15) 
      irdl.operands(%0, %8, %16)
      irdl.results()
    }
    irdl.operation @sort {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::IntegerType>(elementType))) || ((::llvm::isa<::mlir::IndexType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %6 = irdl.any_of(%5) 
      %7 = irdl.all_of(%4, %6) 
      %8 = irdl.all_of(%3, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %14 = irdl.any_of(%13) 
      %15 = irdl.all_of(%12, %14) 
      %16 = irdl.all_of(%11, %15) 
      irdl.operands(%0, %8, %16)
      irdl.results()
    }
    irdl.operation @storage_specifier.init {
      %0 = irdl.c_pred "::llvm::isa<::mlir::sparse_tensor::StorageSpecifierType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::sparse_tensor::StorageSpecifierType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @coordinates_buffer {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
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
      %14 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %15 = irdl.any_of(%14) 
      %16 = irdl.all_of(%13, %15) 
      %17 = irdl.all_of(%12, %16) 
      %18 = irdl.all_of(%9, %17) 
      irdl.operands(%4)
      irdl.results(%18)
    }
    irdl.operation @coordinates {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
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
      %14 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %15 = irdl.any_of(%14) 
      %16 = irdl.all_of(%13, %15) 
      %17 = irdl.all_of(%12, %16) 
      %18 = irdl.all_of(%9, %17) 
      irdl.operands(%4)
      irdl.results(%18)
    }
    irdl.operation @positions {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
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
      %14 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %15 = irdl.any_of(%14) 
      %16 = irdl.all_of(%13, %15) 
      %17 = irdl.all_of(%12, %16) 
      %18 = irdl.all_of(%9, %17) 
      irdl.operands(%4)
      irdl.results(%18)
    }
    irdl.operation @slice.offset {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self) &&   ::mlir::sparse_tensor::getSparseTensorEncoding($_self).isSlice()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @slice.stride {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self) &&   ::mlir::sparse_tensor::getSparseTensorEncoding($_self).isSlice()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @values {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
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
      %14 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %15 = irdl.any_of(%14) 
      %16 = irdl.all_of(%13, %15) 
      %17 = irdl.all_of(%12, %16) 
      %18 = irdl.all_of(%9, %17) 
      irdl.operands(%4)
      irdl.results(%18)
    }
    irdl.operation @unary {
      %0 = irdl.any
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @unpack {
      %0 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %1 = irdl.c_pred "!!::mlir::sparse_tensor::getSparseTensorEncoding($_self)" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %6 = irdl.all_of(%5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %10 = irdl.all_of(%9) 
      %11 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %14 = irdl.all_of(%13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      irdl.operands(%4, %8, %12)
      irdl.results(%16, %20)
    }
    irdl.operation @yield {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
  }
}
