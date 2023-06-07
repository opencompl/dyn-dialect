module {
  irdl.dialect @spirv {
    irdl.operation @SDotAccSat {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.any_of(%4, %20) 
      %22 = irdl.c_pred "$_self.isInteger(8)" 
      %23 = irdl.c_pred "$_self.isInteger(16)" 
      %24 = irdl.c_pred "$_self.isInteger(32)" 
      %25 = irdl.c_pred "$_self.isInteger(64)" 
      %26 = irdl.any_of(%22, %23, %24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %40 = irdl.any_of(%35, %36, %37, %38, %39) 
      %41 = irdl.all_of(%34, %40) 
      %42 = irdl.all_of(%31, %41) 
      %43 = irdl.any_of(%26, %42) 
      %44 = irdl.c_pred "$_self.isInteger(8)" 
      %45 = irdl.c_pred "$_self.isInteger(16)" 
      %46 = irdl.c_pred "$_self.isInteger(32)" 
      %47 = irdl.c_pred "$_self.isInteger(64)" 
      %48 = irdl.any_of(%44, %45, %46, %47) 
      %49 = irdl.c_pred "$_self.isInteger(8)" 
      %50 = irdl.c_pred "$_self.isInteger(16)" 
      %51 = irdl.c_pred "$_self.isInteger(32)" 
      %52 = irdl.c_pred "$_self.isInteger(64)" 
      %53 = irdl.any_of(%49, %50, %51, %52) 
      irdl.operands(%21, %43, %48)
      irdl.results(%53)
    }
    irdl.operation @SDot {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.any_of(%4, %20) 
      %22 = irdl.c_pred "$_self.isInteger(8)" 
      %23 = irdl.c_pred "$_self.isInteger(16)" 
      %24 = irdl.c_pred "$_self.isInteger(32)" 
      %25 = irdl.c_pred "$_self.isInteger(64)" 
      %26 = irdl.any_of(%22, %23, %24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %40 = irdl.any_of(%35, %36, %37, %38, %39) 
      %41 = irdl.all_of(%34, %40) 
      %42 = irdl.all_of(%31, %41) 
      %43 = irdl.any_of(%26, %42) 
      %44 = irdl.c_pred "$_self.isInteger(8)" 
      %45 = irdl.c_pred "$_self.isInteger(16)" 
      %46 = irdl.c_pred "$_self.isInteger(32)" 
      %47 = irdl.c_pred "$_self.isInteger(64)" 
      %48 = irdl.any_of(%44, %45, %46, %47) 
      irdl.operands(%21, %43)
      irdl.results(%48)
    }
    irdl.operation @SUDotAccSat {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.any_of(%4, %20) 
      %22 = irdl.c_pred "$_self.isInteger(8)" 
      %23 = irdl.c_pred "$_self.isInteger(16)" 
      %24 = irdl.c_pred "$_self.isInteger(32)" 
      %25 = irdl.c_pred "$_self.isInteger(64)" 
      %26 = irdl.any_of(%22, %23, %24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %40 = irdl.any_of(%35, %36, %37, %38, %39) 
      %41 = irdl.all_of(%34, %40) 
      %42 = irdl.all_of(%31, %41) 
      %43 = irdl.any_of(%26, %42) 
      %44 = irdl.c_pred "$_self.isInteger(8)" 
      %45 = irdl.c_pred "$_self.isInteger(16)" 
      %46 = irdl.c_pred "$_self.isInteger(32)" 
      %47 = irdl.c_pred "$_self.isInteger(64)" 
      %48 = irdl.any_of(%44, %45, %46, %47) 
      %49 = irdl.c_pred "$_self.isInteger(8)" 
      %50 = irdl.c_pred "$_self.isInteger(16)" 
      %51 = irdl.c_pred "$_self.isInteger(32)" 
      %52 = irdl.c_pred "$_self.isInteger(64)" 
      %53 = irdl.any_of(%49, %50, %51, %52) 
      irdl.operands(%21, %43, %48)
      irdl.results(%53)
    }
    irdl.operation @SUDot {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.any_of(%4, %20) 
      %22 = irdl.c_pred "$_self.isInteger(8)" 
      %23 = irdl.c_pred "$_self.isInteger(16)" 
      %24 = irdl.c_pred "$_self.isInteger(32)" 
      %25 = irdl.c_pred "$_self.isInteger(64)" 
      %26 = irdl.any_of(%22, %23, %24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %40 = irdl.any_of(%35, %36, %37, %38, %39) 
      %41 = irdl.all_of(%34, %40) 
      %42 = irdl.all_of(%31, %41) 
      %43 = irdl.any_of(%26, %42) 
      %44 = irdl.c_pred "$_self.isInteger(8)" 
      %45 = irdl.c_pred "$_self.isInteger(16)" 
      %46 = irdl.c_pred "$_self.isInteger(32)" 
      %47 = irdl.c_pred "$_self.isInteger(64)" 
      %48 = irdl.any_of(%44, %45, %46, %47) 
      irdl.operands(%21, %43)
      irdl.results(%48)
    }
    irdl.operation @UDotAccSat {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.any_of(%4, %20) 
      %22 = irdl.c_pred "$_self.isInteger(8)" 
      %23 = irdl.c_pred "$_self.isInteger(16)" 
      %24 = irdl.c_pred "$_self.isInteger(32)" 
      %25 = irdl.c_pred "$_self.isInteger(64)" 
      %26 = irdl.any_of(%22, %23, %24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %40 = irdl.any_of(%35, %36, %37, %38, %39) 
      %41 = irdl.all_of(%34, %40) 
      %42 = irdl.all_of(%31, %41) 
      %43 = irdl.any_of(%26, %42) 
      %44 = irdl.c_pred "$_self.isInteger(8)" 
      %45 = irdl.c_pred "$_self.isInteger(16)" 
      %46 = irdl.c_pred "$_self.isInteger(32)" 
      %47 = irdl.c_pred "$_self.isInteger(64)" 
      %48 = irdl.any_of(%44, %45, %46, %47) 
      %49 = irdl.c_pred "$_self.isInteger(8)" 
      %50 = irdl.c_pred "$_self.isInteger(16)" 
      %51 = irdl.c_pred "$_self.isInteger(32)" 
      %52 = irdl.c_pred "$_self.isInteger(64)" 
      %53 = irdl.any_of(%49, %50, %51, %52) 
      irdl.operands(%21, %43, %48)
      irdl.results(%53)
    }
    irdl.operation @UDot {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %18 = irdl.any_of(%13, %14, %15, %16, %17) 
      %19 = irdl.all_of(%12, %18) 
      %20 = irdl.all_of(%9, %19) 
      %21 = irdl.any_of(%4, %20) 
      %22 = irdl.c_pred "$_self.isInteger(8)" 
      %23 = irdl.c_pred "$_self.isInteger(16)" 
      %24 = irdl.c_pred "$_self.isInteger(32)" 
      %25 = irdl.c_pred "$_self.isInteger(64)" 
      %26 = irdl.any_of(%22, %23, %24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %40 = irdl.any_of(%35, %36, %37, %38, %39) 
      %41 = irdl.all_of(%34, %40) 
      %42 = irdl.all_of(%31, %41) 
      %43 = irdl.any_of(%26, %42) 
      %44 = irdl.c_pred "$_self.isInteger(8)" 
      %45 = irdl.c_pred "$_self.isInteger(16)" 
      %46 = irdl.c_pred "$_self.isInteger(32)" 
      %47 = irdl.c_pred "$_self.isInteger(64)" 
      %48 = irdl.any_of(%44, %45, %46, %47) 
      irdl.operands(%21, %43)
      irdl.results(%48)
    }
  }
}
