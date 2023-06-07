module {
  irdl.dialect @quant {
    irdl.operation @dcast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<mlir::quant::QuantizedType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %4 = irdl.all_of(%3) 
      %5 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.any_of(%2, %6, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %14 = irdl.c_pred "::llvm::isa<mlir::quant::QuantizedType>($_self)" 
      %15 = irdl.any_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %17 = irdl.all_of(%16) 
      %18 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %24 = irdl.all_of(%22, %23) 
      %25 = irdl.any_of(%15, %19, %24) 
      irdl.operands(%12)
      irdl.results(%25)
    }
    irdl.operation @qcast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<mlir::quant::QuantizedType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %4 = irdl.all_of(%3) 
      %5 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.any_of(%2, %6, %11) 
      %13 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %14 = irdl.c_pred "::llvm::isa<mlir::quant::QuantizedType>($_self)" 
      %15 = irdl.any_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %17 = irdl.all_of(%16) 
      %18 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %24 = irdl.all_of(%22, %23) 
      %25 = irdl.any_of(%15, %19, %24) 
      irdl.operands(%12)
      irdl.results(%25)
    }
    irdl.operation @scast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<mlir::quant::QuantizedType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %4 = irdl.all_of(%3) 
      %5 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.any_of(%2, %6, %11) 
      %13 = irdl.c_pred "$_self.isSignlessInteger()" 
      %14 = irdl.c_pred "::llvm::isa<mlir::quant::QuantizedType>($_self)" 
      %15 = irdl.any_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %17 = irdl.all_of(%16) 
      %18 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %24 = irdl.all_of(%22, %23) 
      %25 = irdl.any_of(%15, %19, %24) 
      %26 = irdl.any_of(%12, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %28 = irdl.c_pred "::llvm::isa<mlir::quant::QuantizedType>($_self)" 
      %29 = irdl.any_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %31 = irdl.all_of(%30) 
      %32 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "[](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.any_of(%29, %33, %38) 
      %40 = irdl.c_pred "$_self.isSignlessInteger()" 
      %41 = irdl.c_pred "::llvm::isa<mlir::quant::QuantizedType>($_self)" 
      %42 = irdl.any_of(%40, %41) 
      %43 = irdl.c_pred "::llvm::isa<::mlir::TensorType>($_self)" 
      %44 = irdl.all_of(%43) 
      %45 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.any_of(%42, %46, %51) 
      %53 = irdl.any_of(%39, %52) 
      irdl.operands(%26)
      irdl.results(%53)
    }
  }
}
