module {
  irdl.dialect @spirv {
    irdl.operation @AccessChain {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands(%0, %5)
      irdl.results(%6)
    }
    irdl.operation @mlir.addressof {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @AtomicAnd {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isInteger(8)" 
      %7 = irdl.c_pred "$_self.isInteger(16)" 
      %8 = irdl.c_pred "$_self.isInteger(32)" 
      %9 = irdl.c_pred "$_self.isInteger(64)" 
      %10 = irdl.any_of(%6, %7, %8, %9) 
      irdl.operands(%0, %5)
      irdl.results(%10)
    }
    irdl.operation @AtomicCompareExchange {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isInteger(8)" 
      %7 = irdl.c_pred "$_self.isInteger(16)" 
      %8 = irdl.c_pred "$_self.isInteger(32)" 
      %9 = irdl.c_pred "$_self.isInteger(64)" 
      %10 = irdl.any_of(%6, %7, %8, %9) 
      %11 = irdl.c_pred "$_self.isInteger(8)" 
      %12 = irdl.c_pred "$_self.isInteger(16)" 
      %13 = irdl.c_pred "$_self.isInteger(32)" 
      %14 = irdl.c_pred "$_self.isInteger(64)" 
      %15 = irdl.any_of(%11, %12, %13, %14) 
      irdl.operands(%0, %5, %10)
      irdl.results(%15)
    }
    irdl.operation @AtomicCompareExchangeWeak {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isInteger(8)" 
      %7 = irdl.c_pred "$_self.isInteger(16)" 
      %8 = irdl.c_pred "$_self.isInteger(32)" 
      %9 = irdl.c_pred "$_self.isInteger(64)" 
      %10 = irdl.any_of(%6, %7, %8, %9) 
      %11 = irdl.c_pred "$_self.isInteger(8)" 
      %12 = irdl.c_pred "$_self.isInteger(16)" 
      %13 = irdl.c_pred "$_self.isInteger(32)" 
      %14 = irdl.c_pred "$_self.isInteger(64)" 
      %15 = irdl.any_of(%11, %12, %13, %14) 
      irdl.operands(%0, %5, %10)
      irdl.results(%15)
    }
    irdl.operation @AtomicExchange {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isF16()" 
      %7 = irdl.c_pred "$_self.isF32()" 
      %8 = irdl.c_pred "$_self.isF64()" 
      %9 = irdl.any_of(%6, %7, %8) 
      %10 = irdl.any_of(%5, %9) 
      %11 = irdl.c_pred "$_self.isInteger(8)" 
      %12 = irdl.c_pred "$_self.isInteger(16)" 
      %13 = irdl.c_pred "$_self.isInteger(32)" 
      %14 = irdl.c_pred "$_self.isInteger(64)" 
      %15 = irdl.any_of(%11, %12, %13, %14) 
      %16 = irdl.c_pred "$_self.isF16()" 
      %17 = irdl.c_pred "$_self.isF32()" 
      %18 = irdl.c_pred "$_self.isF64()" 
      %19 = irdl.any_of(%16, %17, %18) 
      %20 = irdl.any_of(%15, %19) 
      irdl.operands(%0, %10)
      irdl.results(%20)
    }
    irdl.operation @AtomicIAdd {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isInteger(8)" 
      %7 = irdl.c_pred "$_self.isInteger(16)" 
      %8 = irdl.c_pred "$_self.isInteger(32)" 
      %9 = irdl.c_pred "$_self.isInteger(64)" 
      %10 = irdl.any_of(%6, %7, %8, %9) 
      irdl.operands(%0, %5)
      irdl.results(%10)
    }
    irdl.operation @AtomicIDecrement {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      irdl.operands(%0)
      irdl.results(%5)
    }
    irdl.operation @AtomicIIncrement {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      irdl.operands(%0)
      irdl.results(%5)
    }
    irdl.operation @AtomicISub {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isInteger(8)" 
      %7 = irdl.c_pred "$_self.isInteger(16)" 
      %8 = irdl.c_pred "$_self.isInteger(32)" 
      %9 = irdl.c_pred "$_self.isInteger(64)" 
      %10 = irdl.any_of(%6, %7, %8, %9) 
      irdl.operands(%0, %5)
      irdl.results(%10)
    }
    irdl.operation @AtomicOr {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isInteger(8)" 
      %7 = irdl.c_pred "$_self.isInteger(16)" 
      %8 = irdl.c_pred "$_self.isInteger(32)" 
      %9 = irdl.c_pred "$_self.isInteger(64)" 
      %10 = irdl.any_of(%6, %7, %8, %9) 
      irdl.operands(%0, %5)
      irdl.results(%10)
    }
    irdl.operation @AtomicSMax {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isInteger(8)" 
      %7 = irdl.c_pred "$_self.isInteger(16)" 
      %8 = irdl.c_pred "$_self.isInteger(32)" 
      %9 = irdl.c_pred "$_self.isInteger(64)" 
      %10 = irdl.any_of(%6, %7, %8, %9) 
      irdl.operands(%0, %5)
      irdl.results(%10)
    }
    irdl.operation @AtomicSMin {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isInteger(8)" 
      %7 = irdl.c_pred "$_self.isInteger(16)" 
      %8 = irdl.c_pred "$_self.isInteger(32)" 
      %9 = irdl.c_pred "$_self.isInteger(64)" 
      %10 = irdl.any_of(%6, %7, %8, %9) 
      irdl.operands(%0, %5)
      irdl.results(%10)
    }
    irdl.operation @AtomicUMax {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isInteger(8)" 
      %7 = irdl.c_pred "$_self.isInteger(16)" 
      %8 = irdl.c_pred "$_self.isInteger(32)" 
      %9 = irdl.c_pred "$_self.isInteger(64)" 
      %10 = irdl.any_of(%6, %7, %8, %9) 
      irdl.operands(%0, %5)
      irdl.results(%10)
    }
    irdl.operation @AtomicUMin {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isInteger(8)" 
      %7 = irdl.c_pred "$_self.isInteger(16)" 
      %8 = irdl.c_pred "$_self.isInteger(32)" 
      %9 = irdl.c_pred "$_self.isInteger(64)" 
      %10 = irdl.any_of(%6, %7, %8, %9) 
      irdl.operands(%0, %5)
      irdl.results(%10)
    }
    irdl.operation @AtomicXor {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isInteger(8)" 
      %7 = irdl.c_pred "$_self.isInteger(16)" 
      %8 = irdl.c_pred "$_self.isInteger(32)" 
      %9 = irdl.c_pred "$_self.isInteger(64)" 
      %10 = irdl.any_of(%6, %7, %8, %9) 
      irdl.operands(%0, %5)
      irdl.results(%10)
    }
    irdl.operation @BitCount {
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
      irdl.operands(%21)
      irdl.results(%43)
    }
    irdl.operation @BitFieldInsert {
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
      %54 = irdl.c_pred "$_self.isInteger(8)" 
      %55 = irdl.c_pred "$_self.isInteger(16)" 
      %56 = irdl.c_pred "$_self.isInteger(32)" 
      %57 = irdl.c_pred "$_self.isInteger(64)" 
      %58 = irdl.any_of(%54, %55, %56, %57) 
      %59 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %61 = irdl.all_of(%59, %60) 
      %62 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %63 = irdl.all_of(%61, %62) 
      %64 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %66 = irdl.all_of(%64, %65) 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %68 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %69 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %70 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %71 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %72 = irdl.any_of(%67, %68, %69, %70, %71) 
      %73 = irdl.all_of(%66, %72) 
      %74 = irdl.all_of(%63, %73) 
      %75 = irdl.any_of(%58, %74) 
      irdl.operands(%21, %43, %48, %53)
      irdl.results(%75)
    }
    irdl.operation @BitFieldSExtract {
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
      %27 = irdl.c_pred "$_self.isInteger(8)" 
      %28 = irdl.c_pred "$_self.isInteger(16)" 
      %29 = irdl.c_pred "$_self.isInteger(32)" 
      %30 = irdl.c_pred "$_self.isInteger(64)" 
      %31 = irdl.any_of(%27, %28, %29, %30) 
      %32 = irdl.c_pred "$_self.isInteger(8)" 
      %33 = irdl.c_pred "$_self.isInteger(16)" 
      %34 = irdl.c_pred "$_self.isInteger(32)" 
      %35 = irdl.c_pred "$_self.isInteger(64)" 
      %36 = irdl.any_of(%32, %33, %34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %44 = irdl.all_of(%42, %43) 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %50 = irdl.any_of(%45, %46, %47, %48, %49) 
      %51 = irdl.all_of(%44, %50) 
      %52 = irdl.all_of(%41, %51) 
      %53 = irdl.any_of(%36, %52) 
      irdl.operands(%21, %26, %31)
      irdl.results(%53)
    }
    irdl.operation @BitFieldUExtract {
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
      %27 = irdl.c_pred "$_self.isInteger(8)" 
      %28 = irdl.c_pred "$_self.isInteger(16)" 
      %29 = irdl.c_pred "$_self.isInteger(32)" 
      %30 = irdl.c_pred "$_self.isInteger(64)" 
      %31 = irdl.any_of(%27, %28, %29, %30) 
      %32 = irdl.c_pred "$_self.isInteger(8)" 
      %33 = irdl.c_pred "$_self.isInteger(16)" 
      %34 = irdl.c_pred "$_self.isInteger(32)" 
      %35 = irdl.c_pred "$_self.isInteger(64)" 
      %36 = irdl.any_of(%32, %33, %34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %44 = irdl.all_of(%42, %43) 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %50 = irdl.any_of(%45, %46, %47, %48, %49) 
      %51 = irdl.all_of(%44, %50) 
      %52 = irdl.all_of(%41, %51) 
      %53 = irdl.any_of(%36, %52) 
      irdl.operands(%21, %26, %31)
      irdl.results(%53)
    }
    irdl.operation @BitReverse {
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
      irdl.operands(%21)
      irdl.results(%43)
    }
    irdl.operation @Bitcast {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "$_self.isF16()" 
      %6 = irdl.c_pred "$_self.isF32()" 
      %7 = irdl.c_pred "$_self.isF64()" 
      %8 = irdl.any_of(%5, %6, %7) 
      %9 = irdl.any_of(%4, %8) 
      %10 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %11 = irdl.any_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %25 = irdl.any_of(%20, %21, %22, %23, %24) 
      %26 = irdl.all_of(%19, %25) 
      %27 = irdl.all_of(%16, %26) 
      %28 = irdl.any_of(%11, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %30 = irdl.any_of(%28, %29) 
      %31 = irdl.c_pred "$_self.isInteger(8)" 
      %32 = irdl.c_pred "$_self.isInteger(16)" 
      %33 = irdl.c_pred "$_self.isInteger(32)" 
      %34 = irdl.c_pred "$_self.isInteger(64)" 
      %35 = irdl.any_of(%31, %32, %33, %34) 
      %36 = irdl.c_pred "$_self.isF16()" 
      %37 = irdl.c_pred "$_self.isF32()" 
      %38 = irdl.c_pred "$_self.isF64()" 
      %39 = irdl.any_of(%36, %37, %38) 
      %40 = irdl.any_of(%35, %39) 
      %41 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %42 = irdl.any_of(%40, %41) 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %61 = irdl.any_of(%59, %60) 
      irdl.operands(%30)
      irdl.results(%61)
    }
    irdl.operation @BitwiseAnd {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @BitwiseOr {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @BitwiseXor {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @BranchConditional {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %3 = irdl.c_pred "$_self.isInteger(8)" 
      %4 = irdl.c_pred "$_self.isInteger(16)" 
      %5 = irdl.c_pred "$_self.isInteger(32)" 
      %6 = irdl.c_pred "$_self.isInteger(64)" 
      %7 = irdl.any_of(%3, %4, %5, %6) 
      %8 = irdl.c_pred "$_self.isF16()" 
      %9 = irdl.c_pred "$_self.isF32()" 
      %10 = irdl.c_pred "$_self.isF64()" 
      %11 = irdl.any_of(%8, %9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %25 = irdl.any_of(%20, %21, %22, %23, %24) 
      %26 = irdl.all_of(%19, %25) 
      %27 = irdl.all_of(%16, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %35 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %36 = irdl.any_of(%1, %2, %7, %11, %27, %28, %29, %30, %31, %32, %33, %34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %38 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %39 = irdl.c_pred "$_self.isInteger(8)" 
      %40 = irdl.c_pred "$_self.isInteger(16)" 
      %41 = irdl.c_pred "$_self.isInteger(32)" 
      %42 = irdl.c_pred "$_self.isInteger(64)" 
      %43 = irdl.any_of(%39, %40, %41, %42) 
      %44 = irdl.c_pred "$_self.isF16()" 
      %45 = irdl.c_pred "$_self.isF32()" 
      %46 = irdl.c_pred "$_self.isF64()" 
      %47 = irdl.any_of(%44, %45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %55 = irdl.all_of(%53, %54) 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %61 = irdl.any_of(%56, %57, %58, %59, %60) 
      %62 = irdl.all_of(%55, %61) 
      %63 = irdl.all_of(%52, %62) 
      %64 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %65 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %66 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %67 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %70 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %72 = irdl.any_of(%37, %38, %43, %47, %63, %64, %65, %66, %67, %68, %69, %70, %71) 
      irdl.operands(%0, %36, %72)
      irdl.results()
    }
    irdl.operation @Branch {
      %0 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      %7 = irdl.c_pred "$_self.isF16()" 
      %8 = irdl.c_pred "$_self.isF32()" 
      %9 = irdl.c_pred "$_self.isF64()" 
      %10 = irdl.any_of(%7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %24 = irdl.any_of(%19, %20, %21, %22, %23) 
      %25 = irdl.all_of(%18, %24) 
      %26 = irdl.all_of(%15, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %35 = irdl.any_of(%0, %1, %6, %10, %26, %27, %28, %29, %30, %31, %32, %33, %34) 
      irdl.operands(%35)
      irdl.results()
    }
    irdl.operation @CL.ceil {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @CL.cos {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @CL.erf {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @CL.exp {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @CL.fabs {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @CL.fmax {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isF16()" 
      %43 = irdl.c_pred "$_self.isF32()" 
      %44 = irdl.c_pred "$_self.isF64()" 
      %45 = irdl.any_of(%42, %43, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %59 = irdl.any_of(%54, %55, %56, %57, %58) 
      %60 = irdl.all_of(%53, %59) 
      %61 = irdl.all_of(%50, %60) 
      %62 = irdl.any_of(%45, %61) 
      irdl.operands(%20, %41)
      irdl.results(%62)
    }
    irdl.operation @CL.fmin {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isF16()" 
      %43 = irdl.c_pred "$_self.isF32()" 
      %44 = irdl.c_pred "$_self.isF64()" 
      %45 = irdl.any_of(%42, %43, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %59 = irdl.any_of(%54, %55, %56, %57, %58) 
      %60 = irdl.all_of(%53, %59) 
      %61 = irdl.all_of(%50, %60) 
      %62 = irdl.any_of(%45, %61) 
      irdl.operands(%20, %41)
      irdl.results(%62)
    }
    irdl.operation @CL.floor {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @CL.fma {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isF16()" 
      %43 = irdl.c_pred "$_self.isF32()" 
      %44 = irdl.c_pred "$_self.isF64()" 
      %45 = irdl.any_of(%42, %43, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %59 = irdl.any_of(%54, %55, %56, %57, %58) 
      %60 = irdl.all_of(%53, %59) 
      %61 = irdl.all_of(%50, %60) 
      %62 = irdl.any_of(%45, %61) 
      %63 = irdl.c_pred "$_self.isF16()" 
      %64 = irdl.c_pred "$_self.isF32()" 
      %65 = irdl.c_pred "$_self.isF64()" 
      %66 = irdl.any_of(%63, %64, %65) 
      %67 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %68 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %69 = irdl.all_of(%67, %68) 
      %70 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %71 = irdl.all_of(%69, %70) 
      %72 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %73 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %74 = irdl.all_of(%72, %73) 
      %75 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %76 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %77 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %78 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %79 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %80 = irdl.any_of(%75, %76, %77, %78, %79) 
      %81 = irdl.all_of(%74, %80) 
      %82 = irdl.all_of(%71, %81) 
      %83 = irdl.any_of(%66, %82) 
      irdl.operands(%20, %41, %62)
      irdl.results(%83)
    }
    irdl.operation @CL.log {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @CL.pow {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isF16()" 
      %43 = irdl.c_pred "$_self.isF32()" 
      %44 = irdl.c_pred "$_self.isF64()" 
      %45 = irdl.any_of(%42, %43, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %59 = irdl.any_of(%54, %55, %56, %57, %58) 
      %60 = irdl.all_of(%53, %59) 
      %61 = irdl.all_of(%50, %60) 
      %62 = irdl.any_of(%45, %61) 
      irdl.operands(%20, %41)
      irdl.results(%62)
    }
    irdl.operation @CL.printf {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %3 = irdl.c_pred "$_self.isInteger(8)" 
      %4 = irdl.c_pred "$_self.isInteger(16)" 
      %5 = irdl.c_pred "$_self.isInteger(32)" 
      %6 = irdl.c_pred "$_self.isInteger(64)" 
      %7 = irdl.any_of(%3, %4, %5, %6) 
      %8 = irdl.c_pred "$_self.isF16()" 
      %9 = irdl.c_pred "$_self.isF32()" 
      %10 = irdl.c_pred "$_self.isF64()" 
      %11 = irdl.any_of(%8, %9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %25 = irdl.any_of(%20, %21, %22, %23, %24) 
      %26 = irdl.all_of(%19, %25) 
      %27 = irdl.all_of(%16, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %35 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %36 = irdl.any_of(%1, %2, %7, %11, %27, %28, %29, %30, %31, %32, %33, %34, %35) 
      %37 = irdl.c_pred "$_self.isInteger(8)" 
      %38 = irdl.c_pred "$_self.isInteger(16)" 
      %39 = irdl.c_pred "$_self.isInteger(32)" 
      %40 = irdl.c_pred "$_self.isInteger(64)" 
      %41 = irdl.any_of(%37, %38, %39, %40) 
      irdl.operands(%0, %36)
      irdl.results(%41)
    }
    irdl.operation @CL.rint {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @CL.round {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @CL.rsqrt {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @CL.s_abs {
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
      irdl.operands(%21)
      irdl.results(%43)
    }
    irdl.operation @CL.s_max {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @CL.s_min {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @CL.sin {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @CL.sqrt {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @CL.tanh {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @CL.u_max {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @CL.u_min {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @CompositeConstruct {
      %0 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      %7 = irdl.c_pred "$_self.isF16()" 
      %8 = irdl.c_pred "$_self.isF32()" 
      %9 = irdl.c_pred "$_self.isF64()" 
      %10 = irdl.any_of(%7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %24 = irdl.any_of(%19, %20, %21, %22, %23) 
      %25 = irdl.all_of(%18, %24) 
      %26 = irdl.all_of(%15, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %35 = irdl.any_of(%0, %1, %6, %10, %26, %27, %28, %29, %30, %31, %32, %33, %34) 
      %36 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %40 = irdl.all_of(%38, %39) 
      %41 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %49 = irdl.any_of(%44, %45, %46, %47, %48) 
      %50 = irdl.all_of(%43, %49) 
      %51 = irdl.all_of(%40, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %53 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %54 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %55 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %56 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %57 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %58 = irdl.any_of(%51, %52, %53, %54, %55, %56, %57) 
      irdl.operands(%35)
      irdl.results(%58)
    }
    irdl.operation @CompositeExtract {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %13 = irdl.any_of(%8, %9, %10, %11, %12) 
      %14 = irdl.all_of(%7, %13) 
      %15 = irdl.all_of(%4, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %17 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %18 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %19 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %22 = irdl.any_of(%15, %16, %17, %18, %19, %20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %24 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "$_self.isF16()" 
      %31 = irdl.c_pred "$_self.isF32()" 
      %32 = irdl.c_pred "$_self.isF64()" 
      %33 = irdl.any_of(%30, %31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %47 = irdl.any_of(%42, %43, %44, %45, %46) 
      %48 = irdl.all_of(%41, %47) 
      %49 = irdl.all_of(%38, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %51 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %52 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %53 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %54 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %55 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %56 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %57 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %58 = irdl.any_of(%23, %24, %29, %33, %49, %50, %51, %52, %53, %54, %55, %56, %57) 
      irdl.operands(%22)
      irdl.results(%58)
    }
    irdl.operation @CompositeInsert {
      %0 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      %7 = irdl.c_pred "$_self.isF16()" 
      %8 = irdl.c_pred "$_self.isF32()" 
      %9 = irdl.c_pred "$_self.isF64()" 
      %10 = irdl.any_of(%7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %24 = irdl.any_of(%19, %20, %21, %22, %23) 
      %25 = irdl.all_of(%18, %24) 
      %26 = irdl.all_of(%15, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %35 = irdl.any_of(%0, %1, %6, %10, %26, %27, %28, %29, %30, %31, %32, %33, %34) 
      %36 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %38 = irdl.all_of(%36, %37) 
      %39 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %40 = irdl.all_of(%38, %39) 
      %41 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %49 = irdl.any_of(%44, %45, %46, %47, %48) 
      %50 = irdl.all_of(%43, %49) 
      %51 = irdl.all_of(%40, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %53 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %54 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %55 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %56 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %57 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %58 = irdl.any_of(%51, %52, %53, %54, %55, %56, %57) 
      %59 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %61 = irdl.all_of(%59, %60) 
      %62 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %63 = irdl.all_of(%61, %62) 
      %64 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %66 = irdl.all_of(%64, %65) 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %68 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %69 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %70 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %71 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %72 = irdl.any_of(%67, %68, %69, %70, %71) 
      %73 = irdl.all_of(%66, %72) 
      %74 = irdl.all_of(%63, %73) 
      %75 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %76 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %77 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %78 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %79 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %80 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %81 = irdl.any_of(%74, %75, %76, %77, %78, %79, %80) 
      irdl.operands(%35, %58)
      irdl.results(%81)
    }
    irdl.operation @Constant {
      %0 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      %7 = irdl.c_pred "$_self.isF16()" 
      %8 = irdl.c_pred "$_self.isF32()" 
      %9 = irdl.c_pred "$_self.isF64()" 
      %10 = irdl.any_of(%7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %24 = irdl.any_of(%19, %20, %21, %22, %23) 
      %25 = irdl.all_of(%18, %24) 
      %26 = irdl.all_of(%15, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %35 = irdl.any_of(%0, %1, %6, %10, %26, %27, %28, %29, %30, %31, %32, %33, %34) 
      irdl.operands()
      irdl.results(%35)
    }
    irdl.operation @ControlBarrier {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @ConvertFToS {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isInteger(8)" 
      %25 = irdl.c_pred "$_self.isInteger(16)" 
      %26 = irdl.c_pred "$_self.isInteger(32)" 
      %27 = irdl.c_pred "$_self.isInteger(64)" 
      %28 = irdl.any_of(%24, %25, %26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %42 = irdl.any_of(%37, %38, %39, %40, %41) 
      %43 = irdl.all_of(%36, %42) 
      %44 = irdl.all_of(%33, %43) 
      %45 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %46 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.any_of(%28, %44, %47) 
      irdl.operands(%23)
      irdl.results(%48)
    }
    irdl.operation @ConvertFToU {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isInteger(8)" 
      %25 = irdl.c_pred "$_self.isInteger(16)" 
      %26 = irdl.c_pred "$_self.isInteger(32)" 
      %27 = irdl.c_pred "$_self.isInteger(64)" 
      %28 = irdl.any_of(%24, %25, %26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %42 = irdl.any_of(%37, %38, %39, %40, %41) 
      %43 = irdl.all_of(%36, %42) 
      %44 = irdl.all_of(%33, %43) 
      %45 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %46 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.any_of(%28, %44, %47) 
      irdl.operands(%23)
      irdl.results(%48)
    }
    irdl.operation @ConvertPtrToU {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      irdl.operands(%0)
      irdl.results(%5)
    }
    irdl.operation @ConvertSToF {
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
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isF16()" 
      %26 = irdl.c_pred "$_self.isF32()" 
      %27 = irdl.c_pred "$_self.isF64()" 
      %28 = irdl.any_of(%25, %26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %42 = irdl.any_of(%37, %38, %39, %40, %41) 
      %43 = irdl.all_of(%36, %42) 
      %44 = irdl.all_of(%33, %43) 
      %45 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %46 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.any_of(%28, %44, %47) 
      irdl.operands(%24)
      irdl.results(%48)
    }
    irdl.operation @ConvertUToF {
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
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isF16()" 
      %26 = irdl.c_pred "$_self.isF32()" 
      %27 = irdl.c_pred "$_self.isF64()" 
      %28 = irdl.any_of(%25, %26, %27) 
      %29 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %31 = irdl.all_of(%29, %30) 
      %32 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %42 = irdl.any_of(%37, %38, %39, %40, %41) 
      %43 = irdl.all_of(%36, %42) 
      %44 = irdl.all_of(%33, %43) 
      %45 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %46 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.any_of(%28, %44, %47) 
      irdl.operands(%24)
      irdl.results(%48)
    }
    irdl.operation @ConvertUToPtr {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @CopyMemory {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results()
    }
    irdl.operation @EXT.AtomicFAdd {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isF16()" 
      %2 = irdl.c_pred "$_self.isF32()" 
      %3 = irdl.c_pred "$_self.isF64()" 
      %4 = irdl.any_of(%1, %2, %3) 
      %5 = irdl.c_pred "$_self.isF16()" 
      %6 = irdl.c_pred "$_self.isF32()" 
      %7 = irdl.c_pred "$_self.isF64()" 
      %8 = irdl.any_of(%5, %6, %7) 
      irdl.operands(%0, %4)
      irdl.results(%8)
    }
    irdl.operation @EntryPoint {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @ExecutionMode {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @FAdd {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      %48 = irdl.c_pred "$_self.isF16()" 
      %49 = irdl.c_pred "$_self.isF32()" 
      %50 = irdl.c_pred "$_self.isF64()" 
      %51 = irdl.any_of(%48, %49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %65 = irdl.any_of(%60, %61, %62, %63, %64) 
      %66 = irdl.all_of(%59, %65) 
      %67 = irdl.all_of(%56, %66) 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %70 = irdl.all_of(%68, %69) 
      %71 = irdl.any_of(%51, %67, %70) 
      irdl.operands(%23, %47)
      irdl.results(%71)
    }
    irdl.operation @FConvert {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      irdl.operands(%23)
      irdl.results(%47)
    }
    irdl.operation @FDiv {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      %48 = irdl.c_pred "$_self.isF16()" 
      %49 = irdl.c_pred "$_self.isF32()" 
      %50 = irdl.c_pred "$_self.isF64()" 
      %51 = irdl.any_of(%48, %49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %65 = irdl.any_of(%60, %61, %62, %63, %64) 
      %66 = irdl.all_of(%59, %65) 
      %67 = irdl.all_of(%56, %66) 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %70 = irdl.all_of(%68, %69) 
      %71 = irdl.any_of(%51, %67, %70) 
      irdl.operands(%23, %47)
      irdl.results(%71)
    }
    irdl.operation @FMod {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      %48 = irdl.c_pred "$_self.isF16()" 
      %49 = irdl.c_pred "$_self.isF32()" 
      %50 = irdl.c_pred "$_self.isF64()" 
      %51 = irdl.any_of(%48, %49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %65 = irdl.any_of(%60, %61, %62, %63, %64) 
      %66 = irdl.all_of(%59, %65) 
      %67 = irdl.all_of(%56, %66) 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %70 = irdl.all_of(%68, %69) 
      %71 = irdl.any_of(%51, %67, %70) 
      irdl.operands(%23, %47)
      irdl.results(%71)
    }
    irdl.operation @FMul {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      %48 = irdl.c_pred "$_self.isF16()" 
      %49 = irdl.c_pred "$_self.isF32()" 
      %50 = irdl.c_pred "$_self.isF64()" 
      %51 = irdl.any_of(%48, %49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %65 = irdl.any_of(%60, %61, %62, %63, %64) 
      %66 = irdl.all_of(%59, %65) 
      %67 = irdl.all_of(%56, %66) 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %70 = irdl.all_of(%68, %69) 
      %71 = irdl.any_of(%51, %67, %70) 
      irdl.operands(%23, %47)
      irdl.results(%71)
    }
    irdl.operation @FNegate {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      irdl.operands(%23)
      irdl.results(%47)
    }
    irdl.operation @FOrdEqual {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @FOrdGreaterThanEqual {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @FOrdGreaterThan {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @FOrdLessThanEqual {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @FOrdLessThan {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @FOrdNotEqual {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @FRem {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      %48 = irdl.c_pred "$_self.isF16()" 
      %49 = irdl.c_pred "$_self.isF32()" 
      %50 = irdl.c_pred "$_self.isF64()" 
      %51 = irdl.any_of(%48, %49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %65 = irdl.any_of(%60, %61, %62, %63, %64) 
      %66 = irdl.all_of(%59, %65) 
      %67 = irdl.all_of(%56, %66) 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %70 = irdl.all_of(%68, %69) 
      %71 = irdl.any_of(%51, %67, %70) 
      irdl.operands(%23, %47)
      irdl.results(%71)
    }
    irdl.operation @FSub {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %21 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.any_of(%3, %19, %22) 
      %24 = irdl.c_pred "$_self.isF16()" 
      %25 = irdl.c_pred "$_self.isF32()" 
      %26 = irdl.c_pred "$_self.isF64()" 
      %27 = irdl.any_of(%24, %25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %41 = irdl.any_of(%36, %37, %38, %39, %40) 
      %42 = irdl.all_of(%35, %41) 
      %43 = irdl.all_of(%32, %42) 
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %45 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %46 = irdl.all_of(%44, %45) 
      %47 = irdl.any_of(%27, %43, %46) 
      %48 = irdl.c_pred "$_self.isF16()" 
      %49 = irdl.c_pred "$_self.isF32()" 
      %50 = irdl.c_pred "$_self.isF64()" 
      %51 = irdl.any_of(%48, %49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %65 = irdl.any_of(%60, %61, %62, %63, %64) 
      %66 = irdl.all_of(%59, %65) 
      %67 = irdl.all_of(%56, %66) 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %69 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %70 = irdl.all_of(%68, %69) 
      %71 = irdl.any_of(%51, %67, %70) 
      irdl.operands(%23, %47)
      irdl.results(%71)
    }
    irdl.operation @FUnordEqual {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @FUnordGreaterThanEqual {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @FUnordGreaterThan {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @FUnordLessThanEqual {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @FUnordLessThan {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @FUnordNotEqual {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @func {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @FunctionCall {
      %0 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      %7 = irdl.c_pred "$_self.isF16()" 
      %8 = irdl.c_pred "$_self.isF32()" 
      %9 = irdl.c_pred "$_self.isF64()" 
      %10 = irdl.any_of(%7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %24 = irdl.any_of(%19, %20, %21, %22, %23) 
      %25 = irdl.all_of(%18, %24) 
      %26 = irdl.all_of(%15, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %35 = irdl.any_of(%0, %1, %6, %10, %26, %27, %28, %29, %30, %31, %32, %33, %34) 
      %36 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %37 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %38 = irdl.c_pred "$_self.isInteger(8)" 
      %39 = irdl.c_pred "$_self.isInteger(16)" 
      %40 = irdl.c_pred "$_self.isInteger(32)" 
      %41 = irdl.c_pred "$_self.isInteger(64)" 
      %42 = irdl.any_of(%38, %39, %40, %41) 
      %43 = irdl.c_pred "$_self.isF16()" 
      %44 = irdl.c_pred "$_self.isF32()" 
      %45 = irdl.c_pred "$_self.isF64()" 
      %46 = irdl.any_of(%43, %44, %45) 
      %47 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %60 = irdl.any_of(%55, %56, %57, %58, %59) 
      %61 = irdl.all_of(%54, %60) 
      %62 = irdl.all_of(%51, %61) 
      %63 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %64 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %65 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %66 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %67 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %69 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %70 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %71 = irdl.any_of(%36, %37, %42, %46, %62, %63, %64, %65, %66, %67, %68, %69, %70) 
      irdl.operands(%35)
      irdl.results(%71)
    }
    irdl.operation @GL.Acos {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %16 = irdl.any_of(%11, %12, %13, %14, %15) 
      %17 = irdl.all_of(%10, %16) 
      %18 = irdl.all_of(%7, %17) 
      %19 = irdl.any_of(%2, %18) 
      %20 = irdl.c_pred "$_self.isF16()" 
      %21 = irdl.c_pred "$_self.isF32()" 
      %22 = irdl.any_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %36 = irdl.any_of(%31, %32, %33, %34, %35) 
      %37 = irdl.all_of(%30, %36) 
      %38 = irdl.all_of(%27, %37) 
      %39 = irdl.any_of(%22, %38) 
      irdl.operands(%19)
      irdl.results(%39)
    }
    irdl.operation @GL.Asin {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %16 = irdl.any_of(%11, %12, %13, %14, %15) 
      %17 = irdl.all_of(%10, %16) 
      %18 = irdl.all_of(%7, %17) 
      %19 = irdl.any_of(%2, %18) 
      %20 = irdl.c_pred "$_self.isF16()" 
      %21 = irdl.c_pred "$_self.isF32()" 
      %22 = irdl.any_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %36 = irdl.any_of(%31, %32, %33, %34, %35) 
      %37 = irdl.all_of(%30, %36) 
      %38 = irdl.all_of(%27, %37) 
      %39 = irdl.any_of(%22, %38) 
      irdl.operands(%19)
      irdl.results(%39)
    }
    irdl.operation @GL.Atan {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %16 = irdl.any_of(%11, %12, %13, %14, %15) 
      %17 = irdl.all_of(%10, %16) 
      %18 = irdl.all_of(%7, %17) 
      %19 = irdl.any_of(%2, %18) 
      %20 = irdl.c_pred "$_self.isF16()" 
      %21 = irdl.c_pred "$_self.isF32()" 
      %22 = irdl.any_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %36 = irdl.any_of(%31, %32, %33, %34, %35) 
      %37 = irdl.all_of(%30, %36) 
      %38 = irdl.all_of(%27, %37) 
      %39 = irdl.any_of(%22, %38) 
      irdl.operands(%19)
      irdl.results(%39)
    }
    irdl.operation @GL.Ceil {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @GL.Cos {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %16 = irdl.any_of(%11, %12, %13, %14, %15) 
      %17 = irdl.all_of(%10, %16) 
      %18 = irdl.all_of(%7, %17) 
      %19 = irdl.any_of(%2, %18) 
      %20 = irdl.c_pred "$_self.isF16()" 
      %21 = irdl.c_pred "$_self.isF32()" 
      %22 = irdl.any_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %36 = irdl.any_of(%31, %32, %33, %34, %35) 
      %37 = irdl.all_of(%30, %36) 
      %38 = irdl.all_of(%27, %37) 
      %39 = irdl.any_of(%22, %38) 
      irdl.operands(%19)
      irdl.results(%39)
    }
    irdl.operation @GL.Cosh {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %16 = irdl.any_of(%11, %12, %13, %14, %15) 
      %17 = irdl.all_of(%10, %16) 
      %18 = irdl.all_of(%7, %17) 
      %19 = irdl.any_of(%2, %18) 
      %20 = irdl.c_pred "$_self.isF16()" 
      %21 = irdl.c_pred "$_self.isF32()" 
      %22 = irdl.any_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %36 = irdl.any_of(%31, %32, %33, %34, %35) 
      %37 = irdl.all_of(%30, %36) 
      %38 = irdl.all_of(%27, %37) 
      %39 = irdl.any_of(%22, %38) 
      irdl.operands(%19)
      irdl.results(%39)
    }
    irdl.operation @GL.Exp {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %16 = irdl.any_of(%11, %12, %13, %14, %15) 
      %17 = irdl.all_of(%10, %16) 
      %18 = irdl.all_of(%7, %17) 
      %19 = irdl.any_of(%2, %18) 
      %20 = irdl.c_pred "$_self.isF16()" 
      %21 = irdl.c_pred "$_self.isF32()" 
      %22 = irdl.any_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %36 = irdl.any_of(%31, %32, %33, %34, %35) 
      %37 = irdl.all_of(%30, %36) 
      %38 = irdl.all_of(%27, %37) 
      %39 = irdl.any_of(%22, %38) 
      irdl.operands(%19)
      irdl.results(%39)
    }
    irdl.operation @GL.FAbs {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @GL.FClamp {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isF16()" 
      %43 = irdl.c_pred "$_self.isF32()" 
      %44 = irdl.c_pred "$_self.isF64()" 
      %45 = irdl.any_of(%42, %43, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %59 = irdl.any_of(%54, %55, %56, %57, %58) 
      %60 = irdl.all_of(%53, %59) 
      %61 = irdl.all_of(%50, %60) 
      %62 = irdl.any_of(%45, %61) 
      %63 = irdl.c_pred "$_self.isF16()" 
      %64 = irdl.c_pred "$_self.isF32()" 
      %65 = irdl.c_pred "$_self.isF64()" 
      %66 = irdl.any_of(%63, %64, %65) 
      %67 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %68 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %69 = irdl.all_of(%67, %68) 
      %70 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %71 = irdl.all_of(%69, %70) 
      %72 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %73 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %74 = irdl.all_of(%72, %73) 
      %75 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %76 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %77 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %78 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %79 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %80 = irdl.any_of(%75, %76, %77, %78, %79) 
      %81 = irdl.all_of(%74, %80) 
      %82 = irdl.all_of(%71, %81) 
      %83 = irdl.any_of(%66, %82) 
      irdl.operands(%20, %41, %62)
      irdl.results(%83)
    }
    irdl.operation @GL.FMax {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isF16()" 
      %43 = irdl.c_pred "$_self.isF32()" 
      %44 = irdl.c_pred "$_self.isF64()" 
      %45 = irdl.any_of(%42, %43, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %59 = irdl.any_of(%54, %55, %56, %57, %58) 
      %60 = irdl.all_of(%53, %59) 
      %61 = irdl.all_of(%50, %60) 
      %62 = irdl.any_of(%45, %61) 
      irdl.operands(%20, %41)
      irdl.results(%62)
    }
    irdl.operation @GL.FMin {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isF16()" 
      %43 = irdl.c_pred "$_self.isF32()" 
      %44 = irdl.c_pred "$_self.isF64()" 
      %45 = irdl.any_of(%42, %43, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %59 = irdl.any_of(%54, %55, %56, %57, %58) 
      %60 = irdl.all_of(%53, %59) 
      %61 = irdl.all_of(%50, %60) 
      %62 = irdl.any_of(%45, %61) 
      irdl.operands(%20, %41)
      irdl.results(%62)
    }
    irdl.operation @GL.FMix {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isF16()" 
      %43 = irdl.c_pred "$_self.isF32()" 
      %44 = irdl.c_pred "$_self.isF64()" 
      %45 = irdl.any_of(%42, %43, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %59 = irdl.any_of(%54, %55, %56, %57, %58) 
      %60 = irdl.all_of(%53, %59) 
      %61 = irdl.all_of(%50, %60) 
      %62 = irdl.any_of(%45, %61) 
      %63 = irdl.c_pred "$_self.isF16()" 
      %64 = irdl.c_pred "$_self.isF32()" 
      %65 = irdl.c_pred "$_self.isF64()" 
      %66 = irdl.any_of(%63, %64, %65) 
      %67 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %68 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %69 = irdl.all_of(%67, %68) 
      %70 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %71 = irdl.all_of(%69, %70) 
      %72 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %73 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %74 = irdl.all_of(%72, %73) 
      %75 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %76 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %77 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %78 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %79 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %80 = irdl.any_of(%75, %76, %77, %78, %79) 
      %81 = irdl.all_of(%74, %80) 
      %82 = irdl.all_of(%71, %81) 
      %83 = irdl.any_of(%66, %82) 
      irdl.operands(%20, %41, %62)
      irdl.results(%83)
    }
    irdl.operation @GL.FSign {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @GL.FindUMsb {
      %0 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %14 = irdl.any_of(%9, %10, %11, %12, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.all_of(%5, %15) 
      %17 = irdl.any_of(%0, %16) 
      %18 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %32 = irdl.any_of(%27, %28, %29, %30, %31) 
      %33 = irdl.all_of(%26, %32) 
      %34 = irdl.all_of(%23, %33) 
      %35 = irdl.any_of(%18, %34) 
      irdl.operands(%17)
      irdl.results(%35)
    }
    irdl.operation @GL.Floor {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @GL.Fma {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isF16()" 
      %43 = irdl.c_pred "$_self.isF32()" 
      %44 = irdl.c_pred "$_self.isF64()" 
      %45 = irdl.any_of(%42, %43, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %59 = irdl.any_of(%54, %55, %56, %57, %58) 
      %60 = irdl.all_of(%53, %59) 
      %61 = irdl.all_of(%50, %60) 
      %62 = irdl.any_of(%45, %61) 
      %63 = irdl.c_pred "$_self.isF16()" 
      %64 = irdl.c_pred "$_self.isF32()" 
      %65 = irdl.c_pred "$_self.isF64()" 
      %66 = irdl.any_of(%63, %64, %65) 
      %67 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %68 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %69 = irdl.all_of(%67, %68) 
      %70 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %71 = irdl.all_of(%69, %70) 
      %72 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %73 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %74 = irdl.all_of(%72, %73) 
      %75 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %76 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %77 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %78 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %79 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %80 = irdl.any_of(%75, %76, %77, %78, %79) 
      %81 = irdl.all_of(%74, %80) 
      %82 = irdl.all_of(%71, %81) 
      %83 = irdl.any_of(%66, %82) 
      irdl.operands(%20, %41, %62)
      irdl.results(%83)
    }
    irdl.operation @GL.FrexpStruct {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      irdl.operands(%20)
      irdl.results(%21)
    }
    irdl.operation @GL.InverseSqrt {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @GL.Ldexp {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isInteger(8)" 
      %22 = irdl.c_pred "$_self.isInteger(16)" 
      %23 = irdl.c_pred "$_self.isInteger(32)" 
      %24 = irdl.c_pred "$_self.isInteger(64)" 
      %25 = irdl.any_of(%21, %22, %23, %24) 
      %26 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %28 = irdl.all_of(%26, %27) 
      %29 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %33 = irdl.all_of(%31, %32) 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %39 = irdl.any_of(%34, %35, %36, %37, %38) 
      %40 = irdl.all_of(%33, %39) 
      %41 = irdl.all_of(%30, %40) 
      %42 = irdl.any_of(%25, %41) 
      %43 = irdl.c_pred "$_self.isF16()" 
      %44 = irdl.c_pred "$_self.isF32()" 
      %45 = irdl.c_pred "$_self.isF64()" 
      %46 = irdl.any_of(%43, %44, %45) 
      %47 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %60 = irdl.any_of(%55, %56, %57, %58, %59) 
      %61 = irdl.all_of(%54, %60) 
      %62 = irdl.all_of(%51, %61) 
      %63 = irdl.any_of(%46, %62) 
      irdl.operands(%20, %42)
      irdl.results(%63)
    }
    irdl.operation @GL.Log {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %16 = irdl.any_of(%11, %12, %13, %14, %15) 
      %17 = irdl.all_of(%10, %16) 
      %18 = irdl.all_of(%7, %17) 
      %19 = irdl.any_of(%2, %18) 
      %20 = irdl.c_pred "$_self.isF16()" 
      %21 = irdl.c_pred "$_self.isF32()" 
      %22 = irdl.any_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %36 = irdl.any_of(%31, %32, %33, %34, %35) 
      %37 = irdl.all_of(%30, %36) 
      %38 = irdl.all_of(%27, %37) 
      %39 = irdl.any_of(%22, %38) 
      irdl.operands(%19)
      irdl.results(%39)
    }
    irdl.operation @GL.Pow {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %16 = irdl.any_of(%11, %12, %13, %14, %15) 
      %17 = irdl.all_of(%10, %16) 
      %18 = irdl.all_of(%7, %17) 
      %19 = irdl.any_of(%2, %18) 
      %20 = irdl.c_pred "$_self.isF16()" 
      %21 = irdl.c_pred "$_self.isF32()" 
      %22 = irdl.any_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %36 = irdl.any_of(%31, %32, %33, %34, %35) 
      %37 = irdl.all_of(%30, %36) 
      %38 = irdl.all_of(%27, %37) 
      %39 = irdl.any_of(%22, %38) 
      %40 = irdl.c_pred "$_self.isF16()" 
      %41 = irdl.c_pred "$_self.isF32()" 
      %42 = irdl.any_of(%40, %41) 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%19, %39)
      irdl.results(%59)
    }
    irdl.operation @GL.RoundEven {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @GL.Round {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @GL.SAbs {
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
      irdl.operands(%21)
      irdl.results(%43)
    }
    irdl.operation @GL.SClamp {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      %66 = irdl.c_pred "$_self.isInteger(8)" 
      %67 = irdl.c_pred "$_self.isInteger(16)" 
      %68 = irdl.c_pred "$_self.isInteger(32)" 
      %69 = irdl.c_pred "$_self.isInteger(64)" 
      %70 = irdl.any_of(%66, %67, %68, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %72 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %75 = irdl.all_of(%73, %74) 
      %76 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %77 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %78 = irdl.all_of(%76, %77) 
      %79 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %80 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %81 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %82 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %83 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %84 = irdl.any_of(%79, %80, %81, %82, %83) 
      %85 = irdl.all_of(%78, %84) 
      %86 = irdl.all_of(%75, %85) 
      %87 = irdl.any_of(%70, %86) 
      irdl.operands(%21, %43, %65)
      irdl.results(%87)
    }
    irdl.operation @GL.SMax {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @GL.SMin {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @GL.SSign {
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
      irdl.operands(%21)
      irdl.results(%43)
    }
    irdl.operation @GL.Sin {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %16 = irdl.any_of(%11, %12, %13, %14, %15) 
      %17 = irdl.all_of(%10, %16) 
      %18 = irdl.all_of(%7, %17) 
      %19 = irdl.any_of(%2, %18) 
      %20 = irdl.c_pred "$_self.isF16()" 
      %21 = irdl.c_pred "$_self.isF32()" 
      %22 = irdl.any_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %36 = irdl.any_of(%31, %32, %33, %34, %35) 
      %37 = irdl.all_of(%30, %36) 
      %38 = irdl.all_of(%27, %37) 
      %39 = irdl.any_of(%22, %38) 
      irdl.operands(%19)
      irdl.results(%39)
    }
    irdl.operation @GL.Sinh {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %16 = irdl.any_of(%11, %12, %13, %14, %15) 
      %17 = irdl.all_of(%10, %16) 
      %18 = irdl.all_of(%7, %17) 
      %19 = irdl.any_of(%2, %18) 
      %20 = irdl.c_pred "$_self.isF16()" 
      %21 = irdl.c_pred "$_self.isF32()" 
      %22 = irdl.any_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %36 = irdl.any_of(%31, %32, %33, %34, %35) 
      %37 = irdl.all_of(%30, %36) 
      %38 = irdl.all_of(%27, %37) 
      %39 = irdl.any_of(%22, %38) 
      irdl.operands(%19)
      irdl.results(%39)
    }
    irdl.operation @GL.Sqrt {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @GL.Tan {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %16 = irdl.any_of(%11, %12, %13, %14, %15) 
      %17 = irdl.all_of(%10, %16) 
      %18 = irdl.all_of(%7, %17) 
      %19 = irdl.any_of(%2, %18) 
      %20 = irdl.c_pred "$_self.isF16()" 
      %21 = irdl.c_pred "$_self.isF32()" 
      %22 = irdl.any_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %36 = irdl.any_of(%31, %32, %33, %34, %35) 
      %37 = irdl.all_of(%30, %36) 
      %38 = irdl.all_of(%27, %37) 
      %39 = irdl.any_of(%22, %38) 
      irdl.operands(%19)
      irdl.results(%39)
    }
    irdl.operation @GL.Tanh {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %16 = irdl.any_of(%11, %12, %13, %14, %15) 
      %17 = irdl.all_of(%10, %16) 
      %18 = irdl.all_of(%7, %17) 
      %19 = irdl.any_of(%2, %18) 
      %20 = irdl.c_pred "$_self.isF16()" 
      %21 = irdl.c_pred "$_self.isF32()" 
      %22 = irdl.any_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %30 = irdl.all_of(%28, %29) 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %36 = irdl.any_of(%31, %32, %33, %34, %35) 
      %37 = irdl.all_of(%30, %36) 
      %38 = irdl.all_of(%27, %37) 
      %39 = irdl.any_of(%22, %38) 
      irdl.operands(%19)
      irdl.results(%39)
    }
    irdl.operation @GL.UClamp {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      %66 = irdl.c_pred "$_self.isInteger(8)" 
      %67 = irdl.c_pred "$_self.isInteger(16)" 
      %68 = irdl.c_pred "$_self.isInteger(32)" 
      %69 = irdl.c_pred "$_self.isInteger(64)" 
      %70 = irdl.any_of(%66, %67, %68, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %72 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %75 = irdl.all_of(%73, %74) 
      %76 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %77 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %78 = irdl.all_of(%76, %77) 
      %79 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %80 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %81 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %82 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %83 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %84 = irdl.any_of(%79, %80, %81, %82, %83) 
      %85 = irdl.all_of(%78, %84) 
      %86 = irdl.all_of(%75, %85) 
      %87 = irdl.any_of(%70, %86) 
      irdl.operands(%21, %43, %65)
      irdl.results(%87)
    }
    irdl.operation @GL.UMax {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @GL.UMin {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @GenericCastToPtrExplicit {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @GenericCastToPtr {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @GlobalVariable {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @GroupBroadcast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      %7 = irdl.c_pred "$_self.isF16()" 
      %8 = irdl.c_pred "$_self.isF32()" 
      %9 = irdl.c_pred "$_self.isF64()" 
      %10 = irdl.any_of(%7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %24 = irdl.any_of(%19, %20, %21, %22, %23) 
      %25 = irdl.all_of(%18, %24) 
      %26 = irdl.all_of(%15, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %35 = irdl.any_of(%0, %1, %6, %10, %26, %27, %28, %29, %30, %31, %32, %33, %34) 
      %36 = irdl.c_pred "$_self.isInteger(8)" 
      %37 = irdl.c_pred "$_self.isInteger(16)" 
      %38 = irdl.c_pred "$_self.isInteger(32)" 
      %39 = irdl.c_pred "$_self.isInteger(64)" 
      %40 = irdl.any_of(%36, %37, %38, %39) 
      %41 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %43 = irdl.all_of(%41, %42) 
      %44 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %54 = irdl.any_of(%49, %50, %51, %52, %53) 
      %55 = irdl.all_of(%48, %54) 
      %56 = irdl.all_of(%45, %55) 
      %57 = irdl.any_of(%40, %56) 
      %58 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %59 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %60 = irdl.c_pred "$_self.isInteger(8)" 
      %61 = irdl.c_pred "$_self.isInteger(16)" 
      %62 = irdl.c_pred "$_self.isInteger(32)" 
      %63 = irdl.c_pred "$_self.isInteger(64)" 
      %64 = irdl.any_of(%60, %61, %62, %63) 
      %65 = irdl.c_pred "$_self.isF16()" 
      %66 = irdl.c_pred "$_self.isF32()" 
      %67 = irdl.c_pred "$_self.isF64()" 
      %68 = irdl.any_of(%65, %66, %67) 
      %69 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %70 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %71 = irdl.all_of(%69, %70) 
      %72 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %75 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %76 = irdl.all_of(%74, %75) 
      %77 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %78 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %79 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %80 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %81 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %82 = irdl.any_of(%77, %78, %79, %80, %81) 
      %83 = irdl.all_of(%76, %82) 
      %84 = irdl.all_of(%73, %83) 
      %85 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %86 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %87 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %88 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %89 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %90 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %91 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %92 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %93 = irdl.any_of(%58, %59, %64, %68, %84, %85, %86, %87, %88, %89, %90, %91, %92) 
      irdl.operands(%35, %57)
      irdl.results(%93)
    }
    irdl.operation @GroupFAdd {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @GroupFMax {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @GroupFMin {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @KHR.GroupFMul {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      irdl.operands(%20)
      irdl.results(%41)
    }
    irdl.operation @GroupIAdd {
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
      irdl.operands(%21)
      irdl.results(%43)
    }
    irdl.operation @KHR.GroupIMul {
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
      irdl.operands(%21)
      irdl.results(%43)
    }
    irdl.operation @GroupNonUniformBallot {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (((elementType.isSignlessInteger(8))) || ((elementType.isUnsignedInteger(8)))) || (((elementType.isSignlessInteger(16))) || ((elementType.isUnsignedInteger(16)))) || (((elementType.isSignlessInteger(32))) || ((elementType.isUnsignedInteger(32)))) || (((elementType.isSignlessInteger(64))) || ((elementType.isUnsignedInteger(64)))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %10 = irdl.any_of(%9) 
      %11 = irdl.all_of(%8, %10) 
      %12 = irdl.all_of(%5, %11) 
      irdl.operands(%0)
      irdl.results(%12)
    }
    irdl.operation @GroupNonUniformBroadcast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      %7 = irdl.c_pred "$_self.isF16()" 
      %8 = irdl.c_pred "$_self.isF32()" 
      %9 = irdl.c_pred "$_self.isF64()" 
      %10 = irdl.any_of(%7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %24 = irdl.any_of(%19, %20, %21, %22, %23) 
      %25 = irdl.all_of(%18, %24) 
      %26 = irdl.all_of(%15, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %35 = irdl.any_of(%0, %1, %6, %10, %26, %27, %28, %29, %30, %31, %32, %33, %34) 
      %36 = irdl.c_pred "$_self.isInteger(8)" 
      %37 = irdl.c_pred "$_self.isInteger(16)" 
      %38 = irdl.c_pred "$_self.isInteger(32)" 
      %39 = irdl.c_pred "$_self.isInteger(64)" 
      %40 = irdl.any_of(%36, %37, %38, %39) 
      %41 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "$_self.isInteger(8)" 
      %44 = irdl.c_pred "$_self.isInteger(16)" 
      %45 = irdl.c_pred "$_self.isInteger(32)" 
      %46 = irdl.c_pred "$_self.isInteger(64)" 
      %47 = irdl.any_of(%43, %44, %45, %46) 
      %48 = irdl.c_pred "$_self.isF16()" 
      %49 = irdl.c_pred "$_self.isF32()" 
      %50 = irdl.c_pred "$_self.isF64()" 
      %51 = irdl.any_of(%48, %49, %50) 
      %52 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %54 = irdl.all_of(%52, %53) 
      %55 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %62 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %65 = irdl.any_of(%60, %61, %62, %63, %64) 
      %66 = irdl.all_of(%59, %65) 
      %67 = irdl.all_of(%56, %66) 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %69 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %70 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %72 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %73 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %74 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %75 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %76 = irdl.any_of(%41, %42, %47, %51, %67, %68, %69, %70, %71, %72, %73, %74, %75) 
      irdl.operands(%35, %40)
      irdl.results(%76)
    }
    irdl.operation @GroupNonUniformElect {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @GroupNonUniformFAdd {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isInteger(8)" 
      %22 = irdl.c_pred "$_self.isInteger(16)" 
      %23 = irdl.c_pred "$_self.isInteger(32)" 
      %24 = irdl.c_pred "$_self.isInteger(64)" 
      %25 = irdl.any_of(%21, %22, %23, %24) 
      %26 = irdl.c_pred "$_self.isF16()" 
      %27 = irdl.c_pred "$_self.isF32()" 
      %28 = irdl.c_pred "$_self.isF64()" 
      %29 = irdl.any_of(%26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.any_of(%29, %45) 
      irdl.operands(%20, %25)
      irdl.results(%46)
    }
    irdl.operation @GroupNonUniformFMax {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isInteger(8)" 
      %22 = irdl.c_pred "$_self.isInteger(16)" 
      %23 = irdl.c_pred "$_self.isInteger(32)" 
      %24 = irdl.c_pred "$_self.isInteger(64)" 
      %25 = irdl.any_of(%21, %22, %23, %24) 
      %26 = irdl.c_pred "$_self.isF16()" 
      %27 = irdl.c_pred "$_self.isF32()" 
      %28 = irdl.c_pred "$_self.isF64()" 
      %29 = irdl.any_of(%26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.any_of(%29, %45) 
      irdl.operands(%20, %25)
      irdl.results(%46)
    }
    irdl.operation @GroupNonUniformFMin {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isInteger(8)" 
      %22 = irdl.c_pred "$_self.isInteger(16)" 
      %23 = irdl.c_pred "$_self.isInteger(32)" 
      %24 = irdl.c_pred "$_self.isInteger(64)" 
      %25 = irdl.any_of(%21, %22, %23, %24) 
      %26 = irdl.c_pred "$_self.isF16()" 
      %27 = irdl.c_pred "$_self.isF32()" 
      %28 = irdl.c_pred "$_self.isF64()" 
      %29 = irdl.any_of(%26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.any_of(%29, %45) 
      irdl.operands(%20, %25)
      irdl.results(%46)
    }
    irdl.operation @GroupNonUniformFMul {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isInteger(8)" 
      %22 = irdl.c_pred "$_self.isInteger(16)" 
      %23 = irdl.c_pred "$_self.isInteger(32)" 
      %24 = irdl.c_pred "$_self.isInteger(64)" 
      %25 = irdl.any_of(%21, %22, %23, %24) 
      %26 = irdl.c_pred "$_self.isF16()" 
      %27 = irdl.c_pred "$_self.isF32()" 
      %28 = irdl.c_pred "$_self.isF64()" 
      %29 = irdl.any_of(%26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.any_of(%29, %45) 
      irdl.operands(%20, %25)
      irdl.results(%46)
    }
    irdl.operation @GroupNonUniformIAdd {
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
      %27 = irdl.c_pred "$_self.isInteger(8)" 
      %28 = irdl.c_pred "$_self.isInteger(16)" 
      %29 = irdl.c_pred "$_self.isInteger(32)" 
      %30 = irdl.c_pred "$_self.isInteger(64)" 
      %31 = irdl.any_of(%27, %28, %29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %45 = irdl.any_of(%40, %41, %42, %43, %44) 
      %46 = irdl.all_of(%39, %45) 
      %47 = irdl.all_of(%36, %46) 
      %48 = irdl.any_of(%31, %47) 
      irdl.operands(%21, %26)
      irdl.results(%48)
    }
    irdl.operation @GroupNonUniformIMul {
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
      %27 = irdl.c_pred "$_self.isInteger(8)" 
      %28 = irdl.c_pred "$_self.isInteger(16)" 
      %29 = irdl.c_pred "$_self.isInteger(32)" 
      %30 = irdl.c_pred "$_self.isInteger(64)" 
      %31 = irdl.any_of(%27, %28, %29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %45 = irdl.any_of(%40, %41, %42, %43, %44) 
      %46 = irdl.all_of(%39, %45) 
      %47 = irdl.all_of(%36, %46) 
      %48 = irdl.any_of(%31, %47) 
      irdl.operands(%21, %26)
      irdl.results(%48)
    }
    irdl.operation @GroupNonUniformSMax {
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
      %27 = irdl.c_pred "$_self.isInteger(8)" 
      %28 = irdl.c_pred "$_self.isInteger(16)" 
      %29 = irdl.c_pred "$_self.isInteger(32)" 
      %30 = irdl.c_pred "$_self.isInteger(64)" 
      %31 = irdl.any_of(%27, %28, %29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %45 = irdl.any_of(%40, %41, %42, %43, %44) 
      %46 = irdl.all_of(%39, %45) 
      %47 = irdl.all_of(%36, %46) 
      %48 = irdl.any_of(%31, %47) 
      irdl.operands(%21, %26)
      irdl.results(%48)
    }
    irdl.operation @GroupNonUniformSMin {
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
      %27 = irdl.c_pred "$_self.isInteger(8)" 
      %28 = irdl.c_pred "$_self.isInteger(16)" 
      %29 = irdl.c_pred "$_self.isInteger(32)" 
      %30 = irdl.c_pred "$_self.isInteger(64)" 
      %31 = irdl.any_of(%27, %28, %29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %45 = irdl.any_of(%40, %41, %42, %43, %44) 
      %46 = irdl.all_of(%39, %45) 
      %47 = irdl.all_of(%36, %46) 
      %48 = irdl.any_of(%31, %47) 
      irdl.operands(%21, %26)
      irdl.results(%48)
    }
    irdl.operation @GroupNonUniformShuffleDown {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "$_self.isF16()" 
      %6 = irdl.c_pred "$_self.isF32()" 
      %7 = irdl.c_pred "$_self.isF64()" 
      %8 = irdl.any_of(%5, %6, %7) 
      %9 = irdl.any_of(%4, %8) 
      %10 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %11 = irdl.any_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %25 = irdl.any_of(%20, %21, %22, %23, %24) 
      %26 = irdl.all_of(%19, %25) 
      %27 = irdl.all_of(%16, %26) 
      %28 = irdl.any_of(%11, %27) 
      %29 = irdl.c_pred "$_self.isInteger(8)" 
      %30 = irdl.c_pred "$_self.isInteger(16)" 
      %31 = irdl.c_pred "$_self.isInteger(32)" 
      %32 = irdl.c_pred "$_self.isInteger(64)" 
      %33 = irdl.any_of(%29, %30, %31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %35 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %36 = irdl.c_pred "$_self.isInteger(8)" 
      %37 = irdl.c_pred "$_self.isInteger(16)" 
      %38 = irdl.c_pred "$_self.isInteger(32)" 
      %39 = irdl.c_pred "$_self.isInteger(64)" 
      %40 = irdl.any_of(%36, %37, %38, %39) 
      %41 = irdl.c_pred "$_self.isF16()" 
      %42 = irdl.c_pred "$_self.isF32()" 
      %43 = irdl.c_pred "$_self.isF64()" 
      %44 = irdl.any_of(%41, %42, %43) 
      %45 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %58 = irdl.any_of(%53, %54, %55, %56, %57) 
      %59 = irdl.all_of(%52, %58) 
      %60 = irdl.all_of(%49, %59) 
      %61 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %62 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %63 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %64 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %65 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %66 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %67 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %69 = irdl.any_of(%34, %35, %40, %44, %60, %61, %62, %63, %64, %65, %66, %67, %68) 
      irdl.operands(%28, %33)
      irdl.results(%69)
    }
    irdl.operation @GroupNonUniformShuffle {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "$_self.isF16()" 
      %6 = irdl.c_pred "$_self.isF32()" 
      %7 = irdl.c_pred "$_self.isF64()" 
      %8 = irdl.any_of(%5, %6, %7) 
      %9 = irdl.any_of(%4, %8) 
      %10 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %11 = irdl.any_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %25 = irdl.any_of(%20, %21, %22, %23, %24) 
      %26 = irdl.all_of(%19, %25) 
      %27 = irdl.all_of(%16, %26) 
      %28 = irdl.any_of(%11, %27) 
      %29 = irdl.c_pred "$_self.isInteger(8)" 
      %30 = irdl.c_pred "$_self.isInteger(16)" 
      %31 = irdl.c_pred "$_self.isInteger(32)" 
      %32 = irdl.c_pred "$_self.isInteger(64)" 
      %33 = irdl.any_of(%29, %30, %31, %32) 
      %34 = irdl.c_pred "$_self.isInteger(8)" 
      %35 = irdl.c_pred "$_self.isInteger(16)" 
      %36 = irdl.c_pred "$_self.isInteger(32)" 
      %37 = irdl.c_pred "$_self.isInteger(64)" 
      %38 = irdl.any_of(%34, %35, %36, %37) 
      %39 = irdl.c_pred "$_self.isF16()" 
      %40 = irdl.c_pred "$_self.isF32()" 
      %41 = irdl.c_pred "$_self.isF64()" 
      %42 = irdl.any_of(%39, %40, %41) 
      %43 = irdl.any_of(%38, %42) 
      %44 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %45 = irdl.any_of(%43, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %59 = irdl.any_of(%54, %55, %56, %57, %58) 
      %60 = irdl.all_of(%53, %59) 
      %61 = irdl.all_of(%50, %60) 
      %62 = irdl.any_of(%45, %61) 
      irdl.operands(%28, %33)
      irdl.results(%62)
    }
    irdl.operation @GroupNonUniformShuffleUp {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "$_self.isF16()" 
      %6 = irdl.c_pred "$_self.isF32()" 
      %7 = irdl.c_pred "$_self.isF64()" 
      %8 = irdl.any_of(%5, %6, %7) 
      %9 = irdl.any_of(%4, %8) 
      %10 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %11 = irdl.any_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %25 = irdl.any_of(%20, %21, %22, %23, %24) 
      %26 = irdl.all_of(%19, %25) 
      %27 = irdl.all_of(%16, %26) 
      %28 = irdl.any_of(%11, %27) 
      %29 = irdl.c_pred "$_self.isInteger(8)" 
      %30 = irdl.c_pred "$_self.isInteger(16)" 
      %31 = irdl.c_pred "$_self.isInteger(32)" 
      %32 = irdl.c_pred "$_self.isInteger(64)" 
      %33 = irdl.any_of(%29, %30, %31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %35 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %36 = irdl.c_pred "$_self.isInteger(8)" 
      %37 = irdl.c_pred "$_self.isInteger(16)" 
      %38 = irdl.c_pred "$_self.isInteger(32)" 
      %39 = irdl.c_pred "$_self.isInteger(64)" 
      %40 = irdl.any_of(%36, %37, %38, %39) 
      %41 = irdl.c_pred "$_self.isF16()" 
      %42 = irdl.c_pred "$_self.isF32()" 
      %43 = irdl.c_pred "$_self.isF64()" 
      %44 = irdl.any_of(%41, %42, %43) 
      %45 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %58 = irdl.any_of(%53, %54, %55, %56, %57) 
      %59 = irdl.all_of(%52, %58) 
      %60 = irdl.all_of(%49, %59) 
      %61 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %62 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %63 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %64 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %65 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %66 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %67 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %69 = irdl.any_of(%34, %35, %40, %44, %60, %61, %62, %63, %64, %65, %66, %67, %68) 
      irdl.operands(%28, %33)
      irdl.results(%69)
    }
    irdl.operation @GroupNonUniformShuffleXor {
      %0 = irdl.c_pred "$_self.isInteger(8)" 
      %1 = irdl.c_pred "$_self.isInteger(16)" 
      %2 = irdl.c_pred "$_self.isInteger(32)" 
      %3 = irdl.c_pred "$_self.isInteger(64)" 
      %4 = irdl.any_of(%0, %1, %2, %3) 
      %5 = irdl.c_pred "$_self.isF16()" 
      %6 = irdl.c_pred "$_self.isF32()" 
      %7 = irdl.c_pred "$_self.isF64()" 
      %8 = irdl.any_of(%5, %6, %7) 
      %9 = irdl.any_of(%4, %8) 
      %10 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %11 = irdl.any_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %25 = irdl.any_of(%20, %21, %22, %23, %24) 
      %26 = irdl.all_of(%19, %25) 
      %27 = irdl.all_of(%16, %26) 
      %28 = irdl.any_of(%11, %27) 
      %29 = irdl.c_pred "$_self.isInteger(8)" 
      %30 = irdl.c_pred "$_self.isInteger(16)" 
      %31 = irdl.c_pred "$_self.isInteger(32)" 
      %32 = irdl.c_pred "$_self.isInteger(64)" 
      %33 = irdl.any_of(%29, %30, %31, %32) 
      %34 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %35 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %36 = irdl.c_pred "$_self.isInteger(8)" 
      %37 = irdl.c_pred "$_self.isInteger(16)" 
      %38 = irdl.c_pred "$_self.isInteger(32)" 
      %39 = irdl.c_pred "$_self.isInteger(64)" 
      %40 = irdl.any_of(%36, %37, %38, %39) 
      %41 = irdl.c_pred "$_self.isF16()" 
      %42 = irdl.c_pred "$_self.isF32()" 
      %43 = irdl.c_pred "$_self.isF64()" 
      %44 = irdl.any_of(%41, %42, %43) 
      %45 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %58 = irdl.any_of(%53, %54, %55, %56, %57) 
      %59 = irdl.all_of(%52, %58) 
      %60 = irdl.all_of(%49, %59) 
      %61 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %62 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %63 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %64 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %65 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %66 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %67 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %68 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %69 = irdl.any_of(%34, %35, %40, %44, %60, %61, %62, %63, %64, %65, %66, %67, %68) 
      irdl.operands(%28, %33)
      irdl.results(%69)
    }
    irdl.operation @GroupNonUniformUMax {
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
      %27 = irdl.c_pred "$_self.isInteger(8)" 
      %28 = irdl.c_pred "$_self.isInteger(16)" 
      %29 = irdl.c_pred "$_self.isInteger(32)" 
      %30 = irdl.c_pred "$_self.isInteger(64)" 
      %31 = irdl.any_of(%27, %28, %29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %45 = irdl.any_of(%40, %41, %42, %43, %44) 
      %46 = irdl.all_of(%39, %45) 
      %47 = irdl.all_of(%36, %46) 
      %48 = irdl.any_of(%31, %47) 
      irdl.operands(%21, %26)
      irdl.results(%48)
    }
    irdl.operation @GroupNonUniformUMin {
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
      %27 = irdl.c_pred "$_self.isInteger(8)" 
      %28 = irdl.c_pred "$_self.isInteger(16)" 
      %29 = irdl.c_pred "$_self.isInteger(32)" 
      %30 = irdl.c_pred "$_self.isInteger(64)" 
      %31 = irdl.any_of(%27, %28, %29, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %45 = irdl.any_of(%40, %41, %42, %43, %44) 
      %46 = irdl.all_of(%39, %45) 
      %47 = irdl.all_of(%36, %46) 
      %48 = irdl.any_of(%31, %47) 
      irdl.operands(%21, %26)
      irdl.results(%48)
    }
    irdl.operation @GroupSMax {
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
      irdl.operands(%21)
      irdl.results(%43)
    }
    irdl.operation @GroupSMin {
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
      irdl.operands(%21)
      irdl.results(%43)
    }
    irdl.operation @GroupUMax {
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
      irdl.operands(%21)
      irdl.results(%43)
    }
    irdl.operation @GroupUMin {
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
      irdl.operands(%21)
      irdl.results(%43)
    }
    irdl.operation @IAddCarry {
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
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      irdl.operands(%21, %43)
      irdl.results(%44)
    }
    irdl.operation @IAdd {
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
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
    irdl.operation @IEqual {
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
      %44 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %45 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %58 = irdl.any_of(%53, %54, %55, %56, %57) 
      %59 = irdl.all_of(%52, %58) 
      %60 = irdl.all_of(%49, %59) 
      %61 = irdl.any_of(%44, %60) 
      irdl.operands(%21, %43)
      irdl.results(%61)
    }
    irdl.operation @IMul {
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
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
    irdl.operation @INTEL.ConvertBF16ToF {
      %0 = irdl.c_pred "$_self.isSignlessInteger(16)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(16)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %14 = irdl.any_of(%9, %10, %11, %12, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.all_of(%5, %15) 
      %17 = irdl.any_of(%0, %16) 
      %18 = irdl.c_pred "$_self.isF32()" 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %32 = irdl.any_of(%27, %28, %29, %30, %31) 
      %33 = irdl.all_of(%26, %32) 
      %34 = irdl.all_of(%23, %33) 
      %35 = irdl.any_of(%18, %34) 
      irdl.operands(%17)
      irdl.results(%35)
    }
    irdl.operation @INTEL.ConvertFToBF16 {
      %0 = irdl.c_pred "$_self.isF32()" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %14 = irdl.any_of(%9, %10, %11, %12, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.all_of(%5, %15) 
      %17 = irdl.any_of(%0, %16) 
      %18 = irdl.c_pred "$_self.isSignlessInteger(16)" 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(16)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %32 = irdl.any_of(%27, %28, %29, %30, %31) 
      %33 = irdl.all_of(%26, %32) 
      %34 = irdl.all_of(%23, %33) 
      %35 = irdl.any_of(%18, %34) 
      irdl.operands(%17)
      irdl.results(%35)
    }
    irdl.operation @INTEL.JointMatrixLoad {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      irdl.operands(%0, %5)
      irdl.results(%6)
    }
    irdl.operation @INTEL.JointMatrixMad {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      irdl.operands(%0, %1, %2)
      irdl.results(%3)
    }
    irdl.operation @INTEL.JointMatrixStore {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      irdl.operands(%0, %1, %6)
      irdl.results()
    }
    irdl.operation @INTEL.JointMatrixWorkItemLength {
      %0 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @INTEL.SubgroupBlockRead {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %3 = irdl.c_pred "$_self.isInteger(8)" 
      %4 = irdl.c_pred "$_self.isInteger(16)" 
      %5 = irdl.c_pred "$_self.isInteger(32)" 
      %6 = irdl.c_pred "$_self.isInteger(64)" 
      %7 = irdl.any_of(%3, %4, %5, %6) 
      %8 = irdl.c_pred "$_self.isF16()" 
      %9 = irdl.c_pred "$_self.isF32()" 
      %10 = irdl.c_pred "$_self.isF64()" 
      %11 = irdl.any_of(%8, %9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %25 = irdl.any_of(%20, %21, %22, %23, %24) 
      %26 = irdl.all_of(%19, %25) 
      %27 = irdl.all_of(%16, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %35 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %36 = irdl.any_of(%1, %2, %7, %11, %27, %28, %29, %30, %31, %32, %33, %34, %35) 
      irdl.operands(%0)
      irdl.results(%36)
    }
    irdl.operation @INTEL.SubgroupBlockWrite {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %3 = irdl.c_pred "$_self.isInteger(8)" 
      %4 = irdl.c_pred "$_self.isInteger(16)" 
      %5 = irdl.c_pred "$_self.isInteger(32)" 
      %6 = irdl.c_pred "$_self.isInteger(64)" 
      %7 = irdl.any_of(%3, %4, %5, %6) 
      %8 = irdl.c_pred "$_self.isF16()" 
      %9 = irdl.c_pred "$_self.isF32()" 
      %10 = irdl.c_pred "$_self.isF64()" 
      %11 = irdl.any_of(%8, %9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %25 = irdl.any_of(%20, %21, %22, %23, %24) 
      %26 = irdl.all_of(%19, %25) 
      %27 = irdl.all_of(%16, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %35 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %36 = irdl.any_of(%1, %2, %7, %11, %27, %28, %29, %30, %31, %32, %33, %34, %35) 
      irdl.operands(%0, %36)
      irdl.results()
    }
    irdl.operation @INotEqual {
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
      %44 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %45 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %58 = irdl.any_of(%53, %54, %55, %56, %57) 
      %59 = irdl.all_of(%52, %58) 
      %60 = irdl.all_of(%49, %59) 
      %61 = irdl.any_of(%44, %60) 
      irdl.operands(%21, %43)
      irdl.results(%61)
    }
    irdl.operation @ISubBorrow {
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
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      irdl.operands(%21, %43)
      irdl.results(%44)
    }
    irdl.operation @ISub {
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
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
    irdl.operation @ImageDrefGather {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %1 = irdl.c_pred "$_self.isF16()" 
      %2 = irdl.c_pred "$_self.isF32()" 
      %3 = irdl.c_pred "$_self.isF64()" 
      %4 = irdl.any_of(%1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
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
      %22 = irdl.c_pred "$_self.isF16()" 
      %23 = irdl.c_pred "$_self.isF32()" 
      %24 = irdl.c_pred "$_self.isF64()" 
      %25 = irdl.any_of(%22, %23, %24) 
      %26 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %27 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %28 = irdl.c_pred "$_self.isInteger(8)" 
      %29 = irdl.c_pred "$_self.isInteger(16)" 
      %30 = irdl.c_pred "$_self.isInteger(32)" 
      %31 = irdl.c_pred "$_self.isInteger(64)" 
      %32 = irdl.any_of(%28, %29, %30, %31) 
      %33 = irdl.c_pred "$_self.isF16()" 
      %34 = irdl.c_pred "$_self.isF32()" 
      %35 = irdl.c_pred "$_self.isF64()" 
      %36 = irdl.any_of(%33, %34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %44 = irdl.all_of(%42, %43) 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %50 = irdl.any_of(%45, %46, %47, %48, %49) 
      %51 = irdl.all_of(%44, %50) 
      %52 = irdl.all_of(%41, %51) 
      %53 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %54 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %55 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %56 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %57 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %58 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %59 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %60 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %61 = irdl.any_of(%26, %27, %32, %36, %52, %53, %54, %55, %56, %57, %58, %59, %60) 
      %62 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %64 = irdl.all_of(%62, %63) 
      %65 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %66 = irdl.all_of(%64, %65) 
      %67 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %68 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %69 = irdl.all_of(%67, %68) 
      %70 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %71 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %72 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %73 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %74 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %75 = irdl.any_of(%70, %71, %72, %73, %74) 
      %76 = irdl.all_of(%69, %75) 
      %77 = irdl.all_of(%66, %76) 
      irdl.operands(%0, %21, %25, %61)
      irdl.results(%77)
    }
    irdl.operation @Image {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::ImageType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @ImageQuerySize {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::ImageType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %10 = irdl.all_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %19 = irdl.any_of(%14, %15, %16, %17, %18) 
      %20 = irdl.all_of(%13, %19) 
      %21 = irdl.all_of(%10, %20) 
      %22 = irdl.any_of(%5, %21) 
      irdl.operands(%0)
      irdl.results(%22)
    }
    irdl.operation @InBoundsPtrAccessChain {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isInteger(8)" 
      %7 = irdl.c_pred "$_self.isInteger(16)" 
      %8 = irdl.c_pred "$_self.isInteger(32)" 
      %9 = irdl.c_pred "$_self.isInteger(64)" 
      %10 = irdl.any_of(%6, %7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands(%0, %5, %10)
      irdl.results(%11)
    }
    irdl.operation @IsInf {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %22 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %24 = irdl.all_of(%22, %23) 
      %25 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %35 = irdl.any_of(%30, %31, %32, %33, %34) 
      %36 = irdl.all_of(%29, %35) 
      %37 = irdl.all_of(%26, %36) 
      %38 = irdl.any_of(%21, %37) 
      irdl.operands(%20)
      irdl.results(%38)
    }
    irdl.operation @IsNan {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %22 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %24 = irdl.all_of(%22, %23) 
      %25 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %32 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %35 = irdl.any_of(%30, %31, %32, %33, %34) 
      %36 = irdl.all_of(%29, %35) 
      %37 = irdl.all_of(%26, %36) 
      %38 = irdl.any_of(%21, %37) 
      irdl.operands(%20)
      irdl.results(%38)
    }
    irdl.operation @KHR.AssumeTrue {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @KHR.SubgroupBallot {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isInteger(32)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %10 = irdl.any_of(%9) 
      %11 = irdl.all_of(%8, %10) 
      %12 = irdl.all_of(%5, %11) 
      irdl.operands(%0)
      irdl.results(%12)
    }
    irdl.operation @Load {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %3 = irdl.c_pred "$_self.isInteger(8)" 
      %4 = irdl.c_pred "$_self.isInteger(16)" 
      %5 = irdl.c_pred "$_self.isInteger(32)" 
      %6 = irdl.c_pred "$_self.isInteger(64)" 
      %7 = irdl.any_of(%3, %4, %5, %6) 
      %8 = irdl.c_pred "$_self.isF16()" 
      %9 = irdl.c_pred "$_self.isF32()" 
      %10 = irdl.c_pred "$_self.isF64()" 
      %11 = irdl.any_of(%8, %9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %25 = irdl.any_of(%20, %21, %22, %23, %24) 
      %26 = irdl.all_of(%19, %25) 
      %27 = irdl.all_of(%16, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %35 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %36 = irdl.any_of(%1, %2, %7, %11, %27, %28, %29, %30, %31, %32, %33, %34, %35) 
      irdl.operands(%0)
      irdl.results(%36)
    }
    irdl.operation @LogicalAnd {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %14 = irdl.any_of(%9, %10, %11, %12, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.all_of(%5, %15) 
      %17 = irdl.any_of(%0, %16) 
      %18 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %32 = irdl.any_of(%27, %28, %29, %30, %31) 
      %33 = irdl.all_of(%26, %32) 
      %34 = irdl.all_of(%23, %33) 
      %35 = irdl.any_of(%18, %34) 
      %36 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %44 = irdl.all_of(%42, %43) 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %50 = irdl.any_of(%45, %46, %47, %48, %49) 
      %51 = irdl.all_of(%44, %50) 
      %52 = irdl.all_of(%41, %51) 
      %53 = irdl.any_of(%36, %52) 
      irdl.operands(%17, %35)
      irdl.results(%53)
    }
    irdl.operation @LogicalEqual {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %14 = irdl.any_of(%9, %10, %11, %12, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.all_of(%5, %15) 
      %17 = irdl.any_of(%0, %16) 
      %18 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %32 = irdl.any_of(%27, %28, %29, %30, %31) 
      %33 = irdl.all_of(%26, %32) 
      %34 = irdl.all_of(%23, %33) 
      %35 = irdl.any_of(%18, %34) 
      %36 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %44 = irdl.all_of(%42, %43) 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %50 = irdl.any_of(%45, %46, %47, %48, %49) 
      %51 = irdl.all_of(%44, %50) 
      %52 = irdl.all_of(%41, %51) 
      %53 = irdl.any_of(%36, %52) 
      irdl.operands(%17, %35)
      irdl.results(%53)
    }
    irdl.operation @LogicalNotEqual {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %14 = irdl.any_of(%9, %10, %11, %12, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.all_of(%5, %15) 
      %17 = irdl.any_of(%0, %16) 
      %18 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %32 = irdl.any_of(%27, %28, %29, %30, %31) 
      %33 = irdl.all_of(%26, %32) 
      %34 = irdl.all_of(%23, %33) 
      %35 = irdl.any_of(%18, %34) 
      %36 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %44 = irdl.all_of(%42, %43) 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %50 = irdl.any_of(%45, %46, %47, %48, %49) 
      %51 = irdl.all_of(%44, %50) 
      %52 = irdl.all_of(%41, %51) 
      %53 = irdl.any_of(%36, %52) 
      irdl.operands(%17, %35)
      irdl.results(%53)
    }
    irdl.operation @LogicalNot {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %14 = irdl.any_of(%9, %10, %11, %12, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.all_of(%5, %15) 
      %17 = irdl.any_of(%0, %16) 
      %18 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %32 = irdl.any_of(%27, %28, %29, %30, %31) 
      %33 = irdl.all_of(%26, %32) 
      %34 = irdl.all_of(%23, %33) 
      %35 = irdl.any_of(%18, %34) 
      irdl.operands(%17)
      irdl.results(%35)
    }
    irdl.operation @LogicalOr {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %14 = irdl.any_of(%9, %10, %11, %12, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.all_of(%5, %15) 
      %17 = irdl.any_of(%0, %16) 
      %18 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %19 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %26 = irdl.all_of(%24, %25) 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %29 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %30 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %32 = irdl.any_of(%27, %28, %29, %30, %31) 
      %33 = irdl.all_of(%26, %32) 
      %34 = irdl.all_of(%23, %33) 
      %35 = irdl.any_of(%18, %34) 
      %36 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %41 = irdl.all_of(%39, %40) 
      %42 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %44 = irdl.all_of(%42, %43) 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %47 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %48 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %50 = irdl.any_of(%45, %46, %47, %48, %49) 
      %51 = irdl.all_of(%44, %50) 
      %52 = irdl.all_of(%41, %51) 
      %53 = irdl.any_of(%36, %52) 
      irdl.operands(%17, %35)
      irdl.results(%53)
    }
    irdl.operation @mlir.loop {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @MatrixTimesMatrix {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @MatrixTimesScalar {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %2 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isF16()" 
      %6 = irdl.c_pred "$_self.isF32()" 
      %7 = irdl.c_pred "$_self.isF64()" 
      %8 = irdl.any_of(%5, %6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %10 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %11 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF16())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF32())) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isF64()))" 
      %12 = irdl.all_of(%10, %11) 
      %13 = irdl.any_of(%9, %12) 
      irdl.operands(%4, %8)
      irdl.results(%13)
    }
    irdl.operation @MemoryBarrier {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @mlir.merge {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @module {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @NV.CooperativeMatrixLength {
      %0 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @NV.CooperativeMatrixLoad {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      irdl.operands(%0, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @NV.CooperativeMatrixMulAdd {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      irdl.operands(%0, %1, %2)
      irdl.results(%3)
    }
    irdl.operation @NV.CooperativeMatrixStore {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%0, %1, %6, %7)
      irdl.results()
    }
    irdl.operation @Not {
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
      irdl.operands(%21)
      irdl.results(%43)
    }
    irdl.operation @Ordered {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @PtrAccessChain {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isInteger(8)" 
      %2 = irdl.c_pred "$_self.isInteger(16)" 
      %3 = irdl.c_pred "$_self.isInteger(32)" 
      %4 = irdl.c_pred "$_self.isInteger(64)" 
      %5 = irdl.any_of(%1, %2, %3, %4) 
      %6 = irdl.c_pred "$_self.isInteger(8)" 
      %7 = irdl.c_pred "$_self.isInteger(16)" 
      %8 = irdl.c_pred "$_self.isInteger(32)" 
      %9 = irdl.c_pred "$_self.isInteger(64)" 
      %10 = irdl.any_of(%6, %7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands(%0, %5, %10)
      irdl.results(%11)
    }
    irdl.operation @PtrCastToGeneric {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @mlir.referenceof {
      %0 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      %7 = irdl.c_pred "$_self.isF16()" 
      %8 = irdl.c_pred "$_self.isF32()" 
      %9 = irdl.c_pred "$_self.isF64()" 
      %10 = irdl.any_of(%7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %24 = irdl.any_of(%19, %20, %21, %22, %23) 
      %25 = irdl.all_of(%18, %24) 
      %26 = irdl.all_of(%15, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %35 = irdl.any_of(%0, %1, %6, %10, %26, %27, %28, %29, %30, %31, %32, %33, %34) 
      irdl.operands()
      irdl.results(%35)
    }
    irdl.operation @Return {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @ReturnValue {
      %0 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      %7 = irdl.c_pred "$_self.isF16()" 
      %8 = irdl.c_pred "$_self.isF32()" 
      %9 = irdl.c_pred "$_self.isF64()" 
      %10 = irdl.any_of(%7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %24 = irdl.any_of(%19, %20, %21, %22, %23) 
      %25 = irdl.all_of(%18, %24) 
      %26 = irdl.all_of(%15, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %35 = irdl.any_of(%0, %1, %6, %10, %26, %27, %28, %29, %30, %31, %32, %33, %34) 
      irdl.operands(%35)
      irdl.results()
    }
    irdl.operation @SConvert {
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
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      irdl.operands(%24)
      irdl.results(%49)
    }
    irdl.operation @SDiv {
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
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
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
    irdl.operation @SGreaterThanEqual {
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
      %44 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %45 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %58 = irdl.any_of(%53, %54, %55, %56, %57) 
      %59 = irdl.all_of(%52, %58) 
      %60 = irdl.all_of(%49, %59) 
      %61 = irdl.any_of(%44, %60) 
      irdl.operands(%21, %43)
      irdl.results(%61)
    }
    irdl.operation @SGreaterThan {
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
      %44 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %45 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %58 = irdl.any_of(%53, %54, %55, %56, %57) 
      %59 = irdl.all_of(%52, %58) 
      %60 = irdl.all_of(%49, %59) 
      %61 = irdl.any_of(%44, %60) 
      irdl.operands(%21, %43)
      irdl.results(%61)
    }
    irdl.operation @SLessThanEqual {
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
      %44 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %45 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %58 = irdl.any_of(%53, %54, %55, %56, %57) 
      %59 = irdl.all_of(%52, %58) 
      %60 = irdl.all_of(%49, %59) 
      %61 = irdl.any_of(%44, %60) 
      irdl.operands(%21, %43)
      irdl.results(%61)
    }
    irdl.operation @SLessThan {
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
      %44 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %45 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %58 = irdl.any_of(%53, %54, %55, %56, %57) 
      %59 = irdl.all_of(%52, %58) 
      %60 = irdl.all_of(%49, %59) 
      %61 = irdl.any_of(%44, %60) 
      irdl.operands(%21, %43)
      irdl.results(%61)
    }
    irdl.operation @SMod {
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
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
    irdl.operation @SMulExtended {
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
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      irdl.operands(%21, %43)
      irdl.results(%44)
    }
    irdl.operation @SNegate {
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
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      irdl.operands(%24)
      irdl.results(%49)
    }
    irdl.operation @SRem {
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
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
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
    irdl.operation @Select {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %14 = irdl.any_of(%9, %10, %11, %12, %13) 
      %15 = irdl.all_of(%8, %14) 
      %16 = irdl.all_of(%5, %15) 
      %17 = irdl.any_of(%0, %16) 
      %18 = irdl.c_pred "$_self.isInteger(8)" 
      %19 = irdl.c_pred "$_self.isInteger(16)" 
      %20 = irdl.c_pred "$_self.isInteger(32)" 
      %21 = irdl.c_pred "$_self.isInteger(64)" 
      %22 = irdl.any_of(%18, %19, %20, %21) 
      %23 = irdl.c_pred "$_self.isF16()" 
      %24 = irdl.c_pred "$_self.isF32()" 
      %25 = irdl.c_pred "$_self.isF64()" 
      %26 = irdl.any_of(%23, %24, %25) 
      %27 = irdl.any_of(%22, %26) 
      %28 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %29 = irdl.any_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %47 = irdl.any_of(%29, %45, %46) 
      %48 = irdl.c_pred "$_self.isInteger(8)" 
      %49 = irdl.c_pred "$_self.isInteger(16)" 
      %50 = irdl.c_pred "$_self.isInteger(32)" 
      %51 = irdl.c_pred "$_self.isInteger(64)" 
      %52 = irdl.any_of(%48, %49, %50, %51) 
      %53 = irdl.c_pred "$_self.isF16()" 
      %54 = irdl.c_pred "$_self.isF32()" 
      %55 = irdl.c_pred "$_self.isF64()" 
      %56 = irdl.any_of(%53, %54, %55) 
      %57 = irdl.any_of(%52, %56) 
      %58 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %59 = irdl.any_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %64 = irdl.all_of(%62, %63) 
      %65 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %67 = irdl.all_of(%65, %66) 
      %68 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %69 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %70 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %71 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %72 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %73 = irdl.any_of(%68, %69, %70, %71, %72) 
      %74 = irdl.all_of(%67, %73) 
      %75 = irdl.all_of(%64, %74) 
      %76 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %77 = irdl.any_of(%59, %75, %76) 
      %78 = irdl.c_pred "$_self.isInteger(8)" 
      %79 = irdl.c_pred "$_self.isInteger(16)" 
      %80 = irdl.c_pred "$_self.isInteger(32)" 
      %81 = irdl.c_pred "$_self.isInteger(64)" 
      %82 = irdl.any_of(%78, %79, %80, %81) 
      %83 = irdl.c_pred "$_self.isF16()" 
      %84 = irdl.c_pred "$_self.isF32()" 
      %85 = irdl.c_pred "$_self.isF64()" 
      %86 = irdl.any_of(%83, %84, %85) 
      %87 = irdl.any_of(%82, %86) 
      %88 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %89 = irdl.any_of(%87, %88) 
      %90 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %91 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %92 = irdl.all_of(%90, %91) 
      %93 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %94 = irdl.all_of(%92, %93) 
      %95 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %96 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %97 = irdl.all_of(%95, %96) 
      %98 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %99 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %100 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %101 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %102 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %103 = irdl.any_of(%98, %99, %100, %101, %102) 
      %104 = irdl.all_of(%97, %103) 
      %105 = irdl.all_of(%94, %104) 
      %106 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %107 = irdl.any_of(%89, %105, %106) 
      irdl.operands(%17, %47, %77)
      irdl.results(%107)
    }
    irdl.operation @mlir.selection {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @ShiftLeftLogical {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @ShiftRightArithmetic {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @ShiftRightLogical {
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
      %49 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %50 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %51 = irdl.all_of(%49, %50) 
      %52 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %53 = irdl.all_of(%51, %52) 
      %54 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %56 = irdl.all_of(%54, %55) 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %58 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %59 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %60 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %62 = irdl.any_of(%57, %58, %59, %60, %61) 
      %63 = irdl.all_of(%56, %62) 
      %64 = irdl.all_of(%53, %63) 
      %65 = irdl.any_of(%48, %64) 
      irdl.operands(%21, %43)
      irdl.results(%65)
    }
    irdl.operation @SpecConstantComposite {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @SpecConstant {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @SpecConstantOperation {
      %0 = irdl.any
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @Store {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %3 = irdl.c_pred "$_self.isInteger(8)" 
      %4 = irdl.c_pred "$_self.isInteger(16)" 
      %5 = irdl.c_pred "$_self.isInteger(32)" 
      %6 = irdl.c_pred "$_self.isInteger(64)" 
      %7 = irdl.any_of(%3, %4, %5, %6) 
      %8 = irdl.c_pred "$_self.isF16()" 
      %9 = irdl.c_pred "$_self.isF32()" 
      %10 = irdl.c_pred "$_self.isF64()" 
      %11 = irdl.any_of(%8, %9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %16 = irdl.all_of(%14, %15) 
      %17 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %18 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %25 = irdl.any_of(%20, %21, %22, %23, %24) 
      %26 = irdl.all_of(%19, %25) 
      %27 = irdl.all_of(%16, %26) 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %35 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %36 = irdl.any_of(%1, %2, %7, %11, %27, %28, %29, %30, %31, %32, %33, %34, %35) 
      irdl.operands(%0, %36)
      irdl.results()
    }
    irdl.operation @Transpose {
      %0 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @UConvert {
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
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      irdl.operands(%24)
      irdl.results(%49)
    }
    irdl.operation @UDiv {
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
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
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
    irdl.operation @UGreaterThanEqual {
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
      %44 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %45 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %58 = irdl.any_of(%53, %54, %55, %56, %57) 
      %59 = irdl.all_of(%52, %58) 
      %60 = irdl.all_of(%49, %59) 
      %61 = irdl.any_of(%44, %60) 
      irdl.operands(%21, %43)
      irdl.results(%61)
    }
    irdl.operation @UGreaterThan {
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
      %44 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %45 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %58 = irdl.any_of(%53, %54, %55, %56, %57) 
      %59 = irdl.all_of(%52, %58) 
      %60 = irdl.all_of(%49, %59) 
      %61 = irdl.any_of(%44, %60) 
      irdl.operands(%21, %43)
      irdl.results(%61)
    }
    irdl.operation @ULessThanEqual {
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
      %44 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %45 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %58 = irdl.any_of(%53, %54, %55, %56, %57) 
      %59 = irdl.all_of(%52, %58) 
      %60 = irdl.all_of(%49, %59) 
      %61 = irdl.any_of(%44, %60) 
      irdl.operands(%21, %43)
      irdl.results(%61)
    }
    irdl.operation @ULessThan {
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
      %44 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %45 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %46 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %49 = irdl.all_of(%47, %48) 
      %50 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %52 = irdl.all_of(%50, %51) 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %57 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %58 = irdl.any_of(%53, %54, %55, %56, %57) 
      %59 = irdl.all_of(%52, %58) 
      %60 = irdl.all_of(%49, %59) 
      %61 = irdl.any_of(%44, %60) 
      irdl.operands(%21, %43)
      irdl.results(%61)
    }
    irdl.operation @UMod {
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
      %21 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %22 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.any_of(%4, %20, %23) 
      %25 = irdl.c_pred "$_self.isInteger(8)" 
      %26 = irdl.c_pred "$_self.isInteger(16)" 
      %27 = irdl.c_pred "$_self.isInteger(32)" 
      %28 = irdl.c_pred "$_self.isInteger(64)" 
      %29 = irdl.any_of(%25, %26, %27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %43 = irdl.any_of(%38, %39, %40, %41, %42) 
      %44 = irdl.all_of(%37, %43) 
      %45 = irdl.all_of(%34, %44) 
      %46 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %47 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %48 = irdl.all_of(%46, %47) 
      %49 = irdl.any_of(%29, %45, %48) 
      %50 = irdl.c_pred "$_self.isInteger(8)" 
      %51 = irdl.c_pred "$_self.isInteger(16)" 
      %52 = irdl.c_pred "$_self.isInteger(32)" 
      %53 = irdl.c_pred "$_self.isInteger(64)" 
      %54 = irdl.any_of(%50, %51, %52, %53) 
      %55 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %56 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %57 = irdl.all_of(%55, %56) 
      %58 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %59 = irdl.all_of(%57, %58) 
      %60 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %61 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %62 = irdl.all_of(%60, %61) 
      %63 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %64 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %65 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %66 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %67 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %68 = irdl.any_of(%63, %64, %65, %66, %67) 
      %69 = irdl.all_of(%62, %68) 
      %70 = irdl.all_of(%59, %69) 
      %71 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %72 = irdl.c_pred "((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(8))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(16))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(32))) || ((::llvm::cast<::mlir::spirv::CooperativeMatrixNVType>($_self).getElementType().isInteger(64)))" 
      %73 = irdl.all_of(%71, %72) 
      %74 = irdl.any_of(%54, %70, %73) 
      irdl.operands(%24, %49)
      irdl.results(%74)
    }
    irdl.operation @UMulExtended {
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
      %44 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      irdl.operands(%21, %43)
      irdl.results(%44)
    }
    irdl.operation @Undef {
      %0 = irdl.c_pred "::llvm::isa<::mlir::NoneType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %2 = irdl.c_pred "$_self.isInteger(8)" 
      %3 = irdl.c_pred "$_self.isInteger(16)" 
      %4 = irdl.c_pred "$_self.isInteger(32)" 
      %5 = irdl.c_pred "$_self.isInteger(64)" 
      %6 = irdl.any_of(%2, %3, %4, %5) 
      %7 = irdl.c_pred "$_self.isF16()" 
      %8 = irdl.c_pred "$_self.isF32()" 
      %9 = irdl.c_pred "$_self.isF64()" 
      %10 = irdl.any_of(%7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %15 = irdl.all_of(%13, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %20 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %21 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %23 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %24 = irdl.any_of(%19, %20, %21, %22, %23) 
      %25 = irdl.all_of(%18, %24) 
      %26 = irdl.all_of(%15, %25) 
      %27 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      %28 = irdl.c_pred "::llvm::isa<::mlir::spirv::ArrayType>($_self)" 
      %29 = irdl.c_pred "::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)" 
      %30 = irdl.c_pred "::llvm::isa<::mlir::spirv::StructType>($_self)" 
      %31 = irdl.c_pred "::llvm::isa<::mlir::spirv::CooperativeMatrixNVType>($_self)" 
      %32 = irdl.c_pred "::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)" 
      %33 = irdl.c_pred "::llvm::isa<::mlir::spirv::MatrixType>($_self)" 
      %34 = irdl.c_pred "::llvm::isa<::mlir::spirv::SampledImageType>($_self)" 
      %35 = irdl.any_of(%0, %1, %6, %10, %26, %27, %28, %29, %30, %31, %32, %33, %34) 
      irdl.operands()
      irdl.results(%35)
    }
    irdl.operation @Unordered {
      %0 = irdl.c_pred "$_self.isF16()" 
      %1 = irdl.c_pred "$_self.isF32()" 
      %2 = irdl.c_pred "$_self.isF64()" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %5 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %14 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %15 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %17 = irdl.any_of(%12, %13, %14, %15, %16) 
      %18 = irdl.all_of(%11, %17) 
      %19 = irdl.all_of(%8, %18) 
      %20 = irdl.any_of(%3, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %27 = irdl.all_of(%25, %26) 
      %28 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %29 = irdl.all_of(%27, %28) 
      %30 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %31 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %32 = irdl.all_of(%30, %31) 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %35 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %36 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %37 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %38 = irdl.any_of(%33, %34, %35, %36, %37) 
      %39 = irdl.all_of(%32, %38) 
      %40 = irdl.all_of(%29, %39) 
      %41 = irdl.any_of(%24, %40) 
      %42 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %43 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %45 = irdl.all_of(%43, %44) 
      %46 = irdl.c_pred "[](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %47 = irdl.all_of(%45, %46) 
      %48 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %49 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %50 = irdl.all_of(%48, %49) 
      %51 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %52 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %53 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %54 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %55 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %56 = irdl.any_of(%51, %52, %53, %54, %55) 
      %57 = irdl.all_of(%50, %56) 
      %58 = irdl.all_of(%47, %57) 
      %59 = irdl.any_of(%42, %58) 
      irdl.operands(%20, %41)
      irdl.results(%59)
    }
    irdl.operation @Unreachable {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @Variable {
      %0 = irdl.any
      %1 = irdl.c_pred "::llvm::isa<::mlir::spirv::PointerType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @VectorExtractDynamic {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %13 = irdl.any_of(%8, %9, %10, %11, %12) 
      %14 = irdl.all_of(%7, %13) 
      %15 = irdl.all_of(%4, %14) 
      %16 = irdl.c_pred "$_self.isInteger(8)" 
      %17 = irdl.c_pred "$_self.isInteger(16)" 
      %18 = irdl.c_pred "$_self.isInteger(32)" 
      %19 = irdl.c_pred "$_self.isInteger(64)" 
      %20 = irdl.any_of(%16, %17, %18, %19) 
      %21 = irdl.c_pred "$_self.isInteger(8)" 
      %22 = irdl.c_pred "$_self.isInteger(16)" 
      %23 = irdl.c_pred "$_self.isInteger(32)" 
      %24 = irdl.c_pred "$_self.isInteger(64)" 
      %25 = irdl.any_of(%21, %22, %23, %24) 
      %26 = irdl.c_pred "$_self.isF16()" 
      %27 = irdl.c_pred "$_self.isF32()" 
      %28 = irdl.c_pred "$_self.isF64()" 
      %29 = irdl.any_of(%26, %27, %28) 
      %30 = irdl.any_of(%25, %29) 
      %31 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %32 = irdl.any_of(%30, %31) 
      irdl.operands(%15, %20)
      irdl.results(%32)
    }
    irdl.operation @VectorInsertDynamic {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %13 = irdl.any_of(%8, %9, %10, %11, %12) 
      %14 = irdl.all_of(%7, %13) 
      %15 = irdl.all_of(%4, %14) 
      %16 = irdl.c_pred "$_self.isInteger(8)" 
      %17 = irdl.c_pred "$_self.isInteger(16)" 
      %18 = irdl.c_pred "$_self.isInteger(32)" 
      %19 = irdl.c_pred "$_self.isInteger(64)" 
      %20 = irdl.any_of(%16, %17, %18, %19) 
      %21 = irdl.c_pred "$_self.isF16()" 
      %22 = irdl.c_pred "$_self.isF32()" 
      %23 = irdl.c_pred "$_self.isF64()" 
      %24 = irdl.any_of(%21, %22, %23) 
      %25 = irdl.any_of(%20, %24) 
      %26 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %27 = irdl.any_of(%25, %26) 
      %28 = irdl.c_pred "$_self.isInteger(8)" 
      %29 = irdl.c_pred "$_self.isInteger(16)" 
      %30 = irdl.c_pred "$_self.isInteger(32)" 
      %31 = irdl.c_pred "$_self.isInteger(64)" 
      %32 = irdl.any_of(%28, %29, %30, %31) 
      %33 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %34 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %35 = irdl.all_of(%33, %34) 
      %36 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %37 = irdl.all_of(%35, %36) 
      %38 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %39 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %40 = irdl.all_of(%38, %39) 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %45 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %46 = irdl.any_of(%41, %42, %43, %44, %45) 
      %47 = irdl.all_of(%40, %46) 
      %48 = irdl.all_of(%37, %47) 
      irdl.operands(%15, %27, %32)
      irdl.results(%48)
    }
    irdl.operation @VectorShuffle {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %11 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %12 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %13 = irdl.any_of(%8, %9, %10, %11, %12) 
      %14 = irdl.all_of(%7, %13) 
      %15 = irdl.all_of(%4, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %18 = irdl.all_of(%16, %17) 
      %19 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %22 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %23 = irdl.all_of(%21, %22) 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %25 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %29 = irdl.any_of(%24, %25, %26, %27, %28) 
      %30 = irdl.all_of(%23, %29) 
      %31 = irdl.all_of(%20, %30) 
      %32 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %33 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %34 = irdl.all_of(%32, %33) 
      %35 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || (((elementType.isInteger(8))) || ((elementType.isInteger(16))) || ((elementType.isInteger(32))) || ((elementType.isInteger(64)))) || (((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64()))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %36 = irdl.all_of(%34, %35) 
      %37 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %38 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %39 = irdl.all_of(%37, %38) 
      %40 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %41 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %42 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %43 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 8" 
      %44 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 16" 
      %45 = irdl.any_of(%40, %41, %42, %43, %44) 
      %46 = irdl.all_of(%39, %45) 
      %47 = irdl.all_of(%36, %46) 
      irdl.operands(%15, %31)
      irdl.results(%47)
    }
    irdl.operation @VectorTimesScalar {
      %0 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %6 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %11 = irdl.any_of(%8, %9, %10) 
      %12 = irdl.all_of(%7, %11) 
      %13 = irdl.all_of(%4, %12) 
      %14 = irdl.c_pred "$_self.isF16()" 
      %15 = irdl.c_pred "$_self.isF32()" 
      %16 = irdl.c_pred "$_self.isF64()" 
      %17 = irdl.any_of(%14, %15, %16) 
      %18 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %19 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %20 = irdl.all_of(%18, %19) 
      %21 = irdl.c_pred "[](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %22 = irdl.all_of(%20, %21) 
      %23 = irdl.c_pred "::llvm::isa<::mlir::VectorType>($_self)" 
      %24 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getRank() > 0" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 2" 
      %27 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 3" 
      %28 = irdl.c_pred "::llvm::cast<::mlir::VectorType>($_self).getNumElements()\0A                           == 4" 
      %29 = irdl.any_of(%26, %27, %28) 
      %30 = irdl.all_of(%25, %29) 
      %31 = irdl.all_of(%22, %30) 
      irdl.operands(%13, %17)
      irdl.results(%31)
    }
    irdl.operation @mlir.yield {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
  }
}
