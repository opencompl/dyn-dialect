module {
  irdl.dialect @llvm {
    irdl.operation @intr.abs {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %5)
      irdl.results(%6)
    }
    irdl.operation @intr.assume {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @intr.bitreverse {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.bswap {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @call_intrinsic {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @intr.copysign {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.coro.align {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @intr.coro.begin {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %4 = irdl.any_of(%2, %3) 
      %5 = irdl.all_of(%1, %4) 
      %6 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.coro.end {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.all_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.coro.free {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMTokenType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %4 = irdl.any_of(%2, %3) 
      %5 = irdl.all_of(%1, %4) 
      %6 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.coro.id {
      %0 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %4 = irdl.any_of(%2, %3) 
      %5 = irdl.all_of(%1, %4) 
      %6 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %11 = irdl.any_of(%9, %10) 
      %12 = irdl.all_of(%8, %11) 
      %13 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %16 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %17 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %18 = irdl.any_of(%16, %17) 
      %19 = irdl.all_of(%15, %18) 
      %20 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %21 = irdl.all_of(%19, %20) 
      %22 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %7, %14, %21)
      irdl.results(%22)
    }
    irdl.operation @intr.coro.resume {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.all_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %6 = irdl.all_of(%4, %5) 
      irdl.operands(%6)
      irdl.results()
    }
    irdl.operation @intr.coro.save {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.all_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%6)
      irdl.results(%7)
    }
    irdl.operation @intr.coro.size {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @intr.coro.suspend {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMTokenType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @intr.cos {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.ctlz {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %5)
      irdl.results(%6)
    }
    irdl.operation @intr.cttz {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %5)
      irdl.results(%6)
    }
    irdl.operation @intr.ctpop {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.dbg.declare {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @intr.dbg.value {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @intr.debugtrap {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @intr.eh.typeid.for {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.all_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%6)
      irdl.results(%7)
    }
    irdl.operation @intr.exp2 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.exp {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.expect {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "$_self.isSignlessInteger()" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @intr.expect.with.probability {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "$_self.isSignlessInteger()" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @intr.fabs {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.ceil {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.floor {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.fma {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      %15 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9, %14)
      irdl.results(%15)
    }
    irdl.operation @intr.fmuladd {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      %15 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9, %14)
      irdl.results(%15)
    }
    irdl.operation @intr.trunc {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.fshl {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "$_self.isSignlessInteger()" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      %15 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9, %14)
      irdl.results(%15)
    }
    irdl.operation @intr.fshr {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "$_self.isSignlessInteger()" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      %15 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9, %14)
      irdl.results(%15)
    }
    irdl.operation @intr.get.active.lane.mask {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "$_self.isSignlessInteger()" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @intr.is.constant {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @intr.is.fpclass {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %5)
      irdl.results(%6)
    }
    irdl.operation @intr.lifetime.end {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @intr.lifetime.start {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @intr.llrint {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @intr.llround {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @intr.log2 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.log10 {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.log {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.lrint {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @intr.lround {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @intr.masked.load {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      irdl.operands(%0, %3, %4)
      irdl.results(%5)
    }
    irdl.operation @intr.masked.store {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %4 = irdl.all_of(%2, %3) 
      irdl.operands(%0, %1, %4)
      irdl.results()
    }
    irdl.operation @intr.matrix.column.major.load {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger()" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @intr.matrix.column.major.store {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger()" 
      irdl.operands(%0, %1, %2)
      irdl.results()
    }
    irdl.operation @intr.matrix.multiply {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @intr.matrix.transpose {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @intr.maxnum {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.maximum {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.memcpy.inline {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger()" 
      %3 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%0, %1, %2, %3)
      irdl.results()
    }
    irdl.operation @intr.memcpy {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger()" 
      %3 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%0, %1, %2, %3)
      irdl.results()
    }
    irdl.operation @intr.memmove {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger()" 
      %3 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%0, %1, %2, %3)
      irdl.results()
    }
    irdl.operation @intr.memset {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(8)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger()" 
      %3 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%0, %1, %2, %3)
      irdl.results()
    }
    irdl.operation @intr.minnum {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.minimum {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.nearbyint {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.experimental.noalias.scope.decl {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @intr.powi {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %5)
      irdl.results(%6)
    }
    irdl.operation @intr.pow {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.prefetch {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %3 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      irdl.operands(%0, %1, %2, %3)
      irdl.results()
    }
    irdl.operation @intr.rint {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.roundeven {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.round {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.sadd.sat {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.sadd.with.overflow {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.smax {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.smin {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.smul.with.overflow {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.sshl.sat {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.ssub.sat {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.ssub.with.overflow {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.sin {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.sqrt {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4)
      irdl.results(%5)
    }
    irdl.operation @intr.stackrestore {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.all_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %6 = irdl.all_of(%4, %5) 
      irdl.operands(%6)
      irdl.results()
    }
    irdl.operation @intr.stacksave {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @intr.experimental.stepvector {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      irdl.operands()
      irdl.results(%2)
    }
    irdl.operation @intr.threadlocal.address {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @intr.trap {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @intr.uadd.sat {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.uadd.with.overflow {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.ubsantrap {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @intr.umax {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.umin {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.umul.with.overflow {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.ushl.sat {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.usub.sat {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.usub.with.overflow {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger()" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.ashr {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.add {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.and {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.fadd {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.fdiv {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.fmuladd {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %10 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %13 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %11, %12)
      irdl.results(%13)
    }
    irdl.operation @intr.vp.fmul {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.fneg {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @intr.vp.fpext {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @intr.vp.fptosi {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @intr.vp.fptoui {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @intr.vp.fptrunc {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @intr.vp.frem {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.fsub {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.fma {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %10 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %13 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %11, %12)
      irdl.results(%13)
    }
    irdl.operation @intr.vp.inttoptr {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @intr.vp.lshr {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.load {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %4)
      irdl.results(%5)
    }
    irdl.operation @intr.vp.merge {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %3, %4, %5)
      irdl.results(%6)
    }
    irdl.operation @intr.vp.mul {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.or {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.ptrtoint {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @intr.vp.reduce.add {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.vp.reduce.and {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.vp.reduce.fadd {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.vp.reduce.fmax {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.vp.reduce.fmin {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.vp.reduce.fmul {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.vp.reduce.mul {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.vp.reduce.or {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.vp.reduce.smax {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.vp.reduce.smin {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.vp.reduce.umax {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.vp.reduce.umin {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.vp.reduce.xor {
      %0 = irdl.c_pred "$_self.isSignlessInteger()" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %6, %7)
      irdl.results(%8)
    }
    irdl.operation @intr.vp.sdiv {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.sext {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @intr.vp.sitofp {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @intr.vp.srem {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.select {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %3, %4, %5)
      irdl.results(%6)
    }
    irdl.operation @intr.vp.shl {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.store {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      irdl.operands(%0, %1, %4, %5)
      irdl.results()
    }
    irdl.operation @intr.experimental.vp.strided.load {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "$_self.isSignlessInteger()" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %4, %5)
      irdl.results(%6)
    }
    irdl.operation @intr.experimental.vp.strided.store {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger()" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      irdl.operands(%0, %1, %2, %5, %6)
      irdl.results()
    }
    irdl.operation @intr.vp.sub {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.trunc {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @intr.vp.udiv {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.uitofp {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @intr.vp.urem {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.xor {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %10 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %8, %9)
      irdl.results(%10)
    }
    irdl.operation @intr.vp.zext {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(32)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @intr.vacopy {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.all_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %8 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %10 = irdl.any_of(%8, %9) 
      %11 = irdl.all_of(%7, %10) 
      %12 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %13 = irdl.all_of(%11, %12) 
      irdl.operands(%6, %13)
      irdl.results()
    }
    irdl.operation @intr.vaend {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.all_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %6 = irdl.all_of(%4, %5) 
      irdl.operands(%6)
      irdl.results()
    }
    irdl.operation @intr.vastart {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %2 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType().isSignlessInteger(8)" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.all_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == 0" 
      %6 = irdl.all_of(%4, %5) 
      irdl.operands(%6)
      irdl.results()
    }
    irdl.operation @intr.masked.compressstore {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %4 = irdl.all_of(%2, %3) 
      irdl.operands(%0, %1, %4)
      irdl.results()
    }
    irdl.operation @intr.masked.expandload {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3, %4)
      irdl.results(%5)
    }
    irdl.operation @intr.masked.gather {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %4 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      irdl.operands(%2, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @intr.masked.scatter {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %6 = irdl.all_of(%4, %5) 
      irdl.operands(%0, %3, %6)
      irdl.results()
    }
    irdl.operation @intr.vector.extract {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @intr.vector.insert {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @intr.vector.reduce.add {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @intr.vector.reduce.and {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @intr.vector.reduce.fadd {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3)
      irdl.results(%4)
    }
    irdl.operation @intr.vector.reduce.fmax {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @intr.vector.reduce.fmin {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @intr.vector.reduce.fmul {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %3)
      irdl.results(%4)
    }
    irdl.operation @intr.vector.reduce.mul {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @intr.vector.reduce.or {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @intr.vector.reduce.smax {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @intr.vector.reduce.smin {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @intr.vector.reduce.umax {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @intr.vector.reduce.umin {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @intr.vector.reduce.xor {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger()" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @intr.vscale {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
  }
}
