module {
  irdl.dialect @arith {
    irdl.operation @addf {
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
    irdl.operation @addi {
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
    irdl.operation @addui_extended {
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
      %33 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %40 = irdl.all_of(%39) 
      %41 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %42 = irdl.all_of(%40, %41) 
      %43 = irdl.any_of(%33, %38, %42) 
      irdl.operands(%10, %21)
      irdl.results(%32, %43)
    }
    irdl.operation @andi {
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
    irdl.operation @bitcast {
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
      %22 = irdl.any_of(%10, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %24 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<::mlir::IndexType>(elementType))) || ((::llvm::isa<::mlir::FloatType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.any_of(%22, %25) 
      %27 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %34 = irdl.all_of(%33) 
      %35 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.any_of(%27, %32, %36) 
      %38 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %39 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %45 = irdl.all_of(%44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.any_of(%38, %43, %47) 
      %49 = irdl.any_of(%37, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %51 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<::mlir::IndexType>(elementType))) || ((::llvm::isa<::mlir::FloatType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.any_of(%49, %52) 
      irdl.operands(%26)
      irdl.results(%53)
    }
    irdl.operation @ceildivsi {
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
    irdl.operation @ceildivui {
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
    irdl.operation @cmpf {
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
      %22 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %29 = irdl.all_of(%28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.any_of(%22, %27, %31) 
      irdl.operands(%10, %21)
      irdl.results(%32)
    }
    irdl.operation @cmpi {
      %0 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.any_of(%0, %3, %7) 
      %9 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %14 = irdl.all_of(%13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.any_of(%9, %12, %16) 
      %18 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %20 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %23 = irdl.all_of(%22) 
      %24 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.any_of(%18, %21, %25) 
      irdl.operands(%8, %17)
      irdl.results(%26)
    }
    irdl.operation @constant {
      %0 = irdl.any
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @divf {
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
    irdl.operation @divsi {
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
    irdl.operation @divui {
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
    irdl.operation @extf {
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
    irdl.operation @extsi {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.any_of(%0, %3, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger()" 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %14 = irdl.all_of(%13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.any_of(%9, %12, %16) 
      irdl.operands(%8)
      irdl.results(%17)
    }
    irdl.operation @extui {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.any_of(%0, %3, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger()" 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %14 = irdl.all_of(%13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.any_of(%9, %12, %16) 
      irdl.operands(%8)
      irdl.results(%17)
    }
    irdl.operation @fptosi {
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
      %11 = irdl.c_pred "$_self.isSignlessInteger()" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %16 = irdl.all_of(%15) 
      %17 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.any_of(%11, %14, %18) 
      irdl.operands(%10)
      irdl.results(%19)
    }
    irdl.operation @fptoui {
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
      %11 = irdl.c_pred "$_self.isSignlessInteger()" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %16 = irdl.all_of(%15) 
      %17 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.any_of(%11, %14, %18) 
      irdl.operands(%10)
      irdl.results(%19)
    }
    irdl.operation @floordivsi {
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
    irdl.operation @index_cast {
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
      %11 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %12 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<::mlir::IndexType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      %15 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %22 = irdl.all_of(%21) 
      %23 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %24 = irdl.all_of(%22, %23) 
      %25 = irdl.any_of(%15, %20, %24) 
      %26 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %27 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<::mlir::IndexType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %28 = irdl.all_of(%26, %27) 
      %29 = irdl.any_of(%25, %28) 
      irdl.operands(%14)
      irdl.results(%29)
    }
    irdl.operation @index_castui {
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
      %11 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %12 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<::mlir::IndexType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      %15 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %22 = irdl.all_of(%21) 
      %23 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %24 = irdl.all_of(%22, %23) 
      %25 = irdl.any_of(%15, %20, %24) 
      %26 = irdl.c_pred "::llvm::isa<::mlir::MemRefType>($_self)" 
      %27 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<::mlir::IndexType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %28 = irdl.all_of(%26, %27) 
      %29 = irdl.any_of(%25, %28) 
      irdl.operands(%14)
      irdl.results(%29)
    }
    irdl.operation @maxf {
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
    irdl.operation @maxsi {
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
    irdl.operation @maxui {
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
    irdl.operation @minf {
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
    irdl.operation @minsi {
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
    irdl.operation @minui {
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
    irdl.operation @mulf {
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
    irdl.operation @muli {
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
    irdl.operation @mulsi_extended {
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
      %33 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %40 = irdl.all_of(%39) 
      %41 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %42 = irdl.all_of(%40, %41) 
      %43 = irdl.any_of(%33, %38, %42) 
      irdl.operands(%10, %21)
      irdl.results(%32, %43)
    }
    irdl.operation @mului_extended {
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
      %33 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %40 = irdl.all_of(%39) 
      %41 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %42 = irdl.all_of(%40, %41) 
      %43 = irdl.any_of(%33, %38, %42) 
      irdl.operands(%10, %21)
      irdl.results(%32, %43)
    }
    irdl.operation @negf {
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
    irdl.operation @ori {
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
    irdl.operation @remf {
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
    irdl.operation @remsi {
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
    irdl.operation @remui {
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
    irdl.operation @sitofp {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.any_of(%0, %3, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %16 = irdl.all_of(%15) 
      %17 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.any_of(%9, %14, %18) 
      irdl.operands(%8)
      irdl.results(%19)
    }
    irdl.operation @shli {
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
    irdl.operation @shrsi {
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
    irdl.operation @shrui {
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
    irdl.operation @subf {
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
    irdl.operation @subi {
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
    irdl.operation @truncf {
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
    irdl.operation @trunci {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.any_of(%0, %3, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger()" 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %14 = irdl.all_of(%13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.any_of(%9, %12, %16) 
      irdl.operands(%8)
      irdl.results(%17)
    }
    irdl.operation @uitofp {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %5 = irdl.all_of(%4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.any_of(%0, %3, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %16 = irdl.all_of(%15) 
      %17 = irdl.c_pred "[](::mlir::Type elementType) { return (::llvm::isa<::mlir::FloatType>(elementType)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.any_of(%9, %14, %18) 
      irdl.operands(%8)
      irdl.results(%19)
    }
    irdl.operation @xori {
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
    irdl.operation @select {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %7 = irdl.all_of(%6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%0, %5, %9) 
      %11 = irdl.any
      %12 = irdl.any
      %13 = irdl.any
      irdl.operands(%10, %11, %12)
      irdl.results(%13)
    }
  }
}
