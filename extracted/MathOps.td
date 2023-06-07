module {
  irdl.dialect @math {
    irdl.operation @absf {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @absi {
      %0 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @atan2 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      %22 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %29 = irdl.all_of(%28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.any_of(%22, %27, %31) 
      irdl.operands(%10, %21)
      irdl.results(%32)
    }
    irdl.operation @atan {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @cbrt {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @ceil {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @copysign {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      %22 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %29 = irdl.all_of(%28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.any_of(%22, %27, %31) 
      irdl.operands(%10, %21)
      irdl.results(%32)
    }
    irdl.operation @cos {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @ctlz {
      %0 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @cttz {
      %0 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @ctpop {
      %0 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @erf {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @exp2 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @expm1 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @exp {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @fpowi {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      %22 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %29 = irdl.all_of(%28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.any_of(%22, %27, %31) 
      irdl.operands(%10, %21)
      irdl.results(%32)
    }
    irdl.operation @floor {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @fma {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      %22 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %29 = irdl.all_of(%28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.any_of(%22, %27, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %40 = irdl.all_of(%39) 
      %41 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %42 = irdl.all_of(%40, %41) 
      %43 = irdl.any_of(%33, %38, %42) 
      irdl.operands(%10, %21, %32)
      irdl.results(%43)
    }
    irdl.operation @ipowi {
      %0 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      %22 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %29 = irdl.all_of(%28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.any_of(%22, %27, %31) 
      irdl.operands(%10, %21)
      irdl.results(%32)
    }
    irdl.operation @log1p {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @log2 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @log10 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @log {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @powf {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      %22 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %29 = irdl.all_of(%28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.any_of(%22, %27, %31) 
      irdl.operands(%10, %21)
      irdl.results(%32)
    }
    irdl.operation @roundeven {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @round {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @rsqrt {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @sin {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @sqrt {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @tan {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @tanh {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
    irdl.operation @trunc {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %18 = irdl.all_of(%17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.any_of(%11, %16, %20) 
      irdl.operands(%10)
      irdl.results(%21)
    }
  }
}
