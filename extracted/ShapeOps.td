module {
  irdl.dialect @shape {
    irdl.operation @add {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.any_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %8 = irdl.any_of(%6, %7) 
      irdl.operands(%2, %5)
      irdl.results(%8)
    }
    irdl.operation @any {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %11 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %14 = irdl.any_of(%13) 
      %15 = irdl.all_of(%12, %14) 
      %16 = irdl.all_of(%11, %15) 
      %17 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.any_of(%10, %18) 
      irdl.operands(%9)
      irdl.results(%19)
    }
    irdl.operation @assuming_all {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::WitnessType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::shape::WitnessType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @assuming {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::WitnessType>($_self)" 
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @assuming_yield {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @broadcast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %11 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %14 = irdl.any_of(%13) 
      %15 = irdl.all_of(%12, %14) 
      %16 = irdl.all_of(%11, %15) 
      %17 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.any_of(%10, %18) 
      irdl.operands(%9)
      irdl.results(%19)
    }
    irdl.operation @concat {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %11 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %14 = irdl.any_of(%13) 
      %15 = irdl.all_of(%12, %14) 
      %16 = irdl.all_of(%11, %15) 
      %17 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.any_of(%10, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %21 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %24 = irdl.any_of(%23) 
      %25 = irdl.all_of(%22, %24) 
      %26 = irdl.all_of(%21, %25) 
      %27 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %28 = irdl.all_of(%26, %27) 
      %29 = irdl.any_of(%20, %28) 
      irdl.operands(%9, %19)
      irdl.results(%29)
    }
    irdl.operation @const_shape {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      irdl.operands()
      irdl.results(%9)
    }
    irdl.operation @const_size {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @const_witness {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::WitnessType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @cstr_broadcastable {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::shape::WitnessType>($_self)" 
      irdl.operands(%9)
      irdl.results(%10)
    }
    irdl.operation @cstr_eq {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::shape::WitnessType>($_self)" 
      irdl.operands(%9)
      irdl.results(%10)
    }
    irdl.operation @cstr_require {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::shape::WitnessType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @debug_print {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %5 = irdl.any_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @dim {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ShapedType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.any_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %8 = irdl.any_of(%6, %7) 
      irdl.operands(%2, %5)
      irdl.results(%8)
    }
    irdl.operation @div {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.any_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %8 = irdl.any_of(%6, %7) 
      irdl.operands(%2, %5)
      irdl.results(%8)
    }
    irdl.operation @from_extent_tensor {
      %0 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %3 = irdl.any_of(%2) 
      %4 = irdl.all_of(%1, %3) 
      %5 = irdl.all_of(%0, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      irdl.operands(%7)
      irdl.results(%8)
    }
    irdl.operation @from_extents {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @func {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @function_library {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @get_extent {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %11 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %12 = irdl.any_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %14 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %15 = irdl.any_of(%13, %14) 
      irdl.operands(%9, %12)
      irdl.results(%15)
    }
    irdl.operation @index_to_size {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @is_broadcastable {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      %10 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%9)
      irdl.results(%10)
    }
    irdl.operation @max {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %5 = irdl.any_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %8 = irdl.any_of(%6, %7) 
      irdl.operands(%2, %5)
      irdl.results(%8)
    }
    irdl.operation @meet {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %7 = irdl.any_of(%6) 
      %8 = irdl.all_of(%5, %7) 
      %9 = irdl.all_of(%4, %8) 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.any_of(%3, %11) 
      %13 = irdl.any_of(%2, %12) 
      %14 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %15 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %16 = irdl.any_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %18 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %19 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %21 = irdl.any_of(%20) 
      %22 = irdl.all_of(%19, %21) 
      %23 = irdl.all_of(%18, %22) 
      %24 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.any_of(%17, %25) 
      %27 = irdl.any_of(%16, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %30 = irdl.any_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %35 = irdl.any_of(%34) 
      %36 = irdl.all_of(%33, %35) 
      %37 = irdl.all_of(%32, %36) 
      %38 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.any_of(%31, %39) 
      %41 = irdl.any_of(%30, %40) 
      irdl.operands(%13, %27)
      irdl.results(%41)
    }
    irdl.operation @min {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %5 = irdl.any_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %8 = irdl.any_of(%6, %7) 
      irdl.operands(%2, %5)
      irdl.results(%8)
    }
    irdl.operation @mul {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.any_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %8 = irdl.any_of(%6, %7) 
      irdl.operands(%2, %5)
      irdl.results(%8)
    }
    irdl.operation @num_elements {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %11 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %12 = irdl.any_of(%10, %11) 
      irdl.operands(%9)
      irdl.results(%12)
    }
    irdl.operation @rank {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %11 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %12 = irdl.any_of(%10, %11) 
      irdl.operands(%9)
      irdl.results(%12)
    }
    irdl.operation @reduce {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      %10 = irdl.any
      %11 = irdl.any
      irdl.operands(%9, %10)
      irdl.results(%11)
    }
    irdl.operation @return {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @shape_eq {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      %10 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%9)
      irdl.results(%10)
    }
    irdl.operation @shape_of {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ShapedType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::shape::ValueShapeType>($_self)" 
      %4 = irdl.any_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %8 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%6, %10) 
      %12 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%5, %13) 
      irdl.operands(%4)
      irdl.results(%14)
    }
    irdl.operation @size_to_index {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @split_at {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::shape::SizeType>($_self)" 
      %11 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %12 = irdl.any_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %14 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %17 = irdl.any_of(%16) 
      %18 = irdl.all_of(%15, %17) 
      %19 = irdl.all_of(%14, %18) 
      %20 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.any_of(%13, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %24 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %27 = irdl.any_of(%26) 
      %28 = irdl.all_of(%25, %27) 
      %29 = irdl.all_of(%24, %28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.any_of(%23, %31) 
      irdl.operands(%9, %12)
      irdl.results(%22, %32)
    }
    irdl.operation @to_extent_tensor {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %4 = irdl.any_of(%3) 
      %5 = irdl.all_of(%2, %4) 
      %6 = irdl.all_of(%1, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%0, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %11 = irdl.all_of(%10) 
      %12 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %13 = irdl.all_of(%11, %12) 
      irdl.operands(%9)
      irdl.results(%13)
    }
    irdl.operation @value_as_shape {
      %0 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %3 = irdl.any_of(%2) 
      %4 = irdl.all_of(%1, %3) 
      %5 = irdl.all_of(%0, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::IntegerType>(elementType))) || ((::llvm::isa<::mlir::IndexType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::shape::ValueShapeType>($_self)" 
      %9 = irdl.any_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %11 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %14 = irdl.any_of(%13) 
      %15 = irdl.all_of(%12, %14) 
      %16 = irdl.all_of(%11, %15) 
      %17 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.any_of(%10, %18) 
      irdl.operands(%9)
      irdl.results(%19)
    }
    irdl.operation @value_of {
      %0 = irdl.c_pred "::llvm::isa<::mlir::shape::ValueShapeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::ShapedType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      irdl.operands(%0)
      irdl.results(%3)
    }
    irdl.operation @with_shape {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ShapedType>($_self)" 
      %1 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::shape::ValueShapeType>($_self)" 
      %4 = irdl.any_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::shape::ShapeType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).hasRank()" 
      %8 = irdl.c_pred "::llvm::cast<::mlir::ShapedType>($_self).getRank()\0A                         == 1" 
      %9 = irdl.any_of(%8) 
      %10 = irdl.all_of(%7, %9) 
      %11 = irdl.all_of(%6, %10) 
      %12 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::IndexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%5, %13) 
      %15 = irdl.c_pred "::llvm::isa<::mlir::shape::ValueShapeType>($_self)" 
      irdl.operands(%4, %14)
      irdl.results(%15)
    }
    irdl.operation @yield {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
  }
}
