module {
  irdl.dialect @llvm {
    irdl.operation @ashr {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @access_group {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @add {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @addrspacecast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @mlir.addressof {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @alias_scope_domain {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @alias_scope {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @alloca {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @and {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @cmpxchg {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %2 = irdl.c_pred "((::llvm::isa<::mlir::IntegerType>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType()))) || ((::llvm::isa<::mlir::LLVM::LLVMPointerType>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType())))" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.all_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %7 = irdl.any_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %9 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %10 = irdl.any_of(%8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMStructType>($_self)" 
      irdl.operands(%4, %7, %10)
      irdl.results(%11)
    }
    irdl.operation @atomicrmw {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %2 = irdl.c_pred "((::mlir::LLVM::isCompatibleFloatingPointType(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType()))) || ((::llvm::isa<::mlir::IntegerType>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType())))" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.all_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %7 = irdl.any_of(%5, %6) 
      %8 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %9 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %10 = irdl.any_of(%8, %9) 
      irdl.operands(%4, %7)
      irdl.results(%10)
    }
    irdl.operation @bitcast {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "!((::llvm::isa<::mlir::LLVM::LLVMStructType, ::mlir::LLVM::LLVMArrayType>($_self)))" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %4 = irdl.c_pred "!((::llvm::isa<::mlir::LLVM::LLVMStructType, ::mlir::LLVM::LLVMArrayType>($_self)))" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @br {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @call {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @cond_br {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2)
      irdl.results()
    }
    irdl.operation @mlir.constant {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @extractelement {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @extractvalue {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMStructType, ::mlir::LLVM::LLVMArrayType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @fadd {
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
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @fcmp {
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
      %10 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @fdiv {
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
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @fmul {
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
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @fneg {
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
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @fpext {
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
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @fptosi {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @fptoui {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @fptrunc {
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
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @frem {
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
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @fsub {
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
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @fence {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @freeze {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @getelementptr {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @mlir.global_ctors {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @mlir.global_dtors {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @mlir.global {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @icmp {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.any_of(%4, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %12 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %13 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %14 = irdl.all_of(%12, %13) 
      %15 = irdl.any_of(%11, %14) 
      %16 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %17 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %18 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %19 = irdl.all_of(%17, %18) 
      %20 = irdl.any_of(%16, %19) 
      %21 = irdl.any_of(%15, %20) 
      %22 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %23 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %24 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %25 = irdl.all_of(%23, %24) 
      %26 = irdl.any_of(%22, %25) 
      irdl.operands(%10, %21)
      irdl.results(%26)
    }
    irdl.operation @inline_asm {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @insertelement {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "!::llvm::isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>($_self)" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      irdl.operands(%0, %3, %4)
      irdl.results(%5)
    }
    irdl.operation @insertvalue {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMStructType, ::mlir::LLVM::LLVMArrayType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "!::llvm::isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>($_self)" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMStructType, ::mlir::LLVM::LLVMArrayType>($_self)" 
      irdl.operands(%0, %3)
      irdl.results(%4)
    }
    irdl.operation @inttoptr {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @invoke {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %3 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0, %1, %2)
      irdl.results(%3)
    }
    irdl.operation @func {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @lshr {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @landingpad {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @load {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %2 = irdl.c_pred "((((::mlir::LLVM::isCompatibleOuterType(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType()))) && ((!::llvm::isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType())))) && (!(((::llvm::isa<::mlir::LLVM::LLVMStructType>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType()))) && ((::llvm::cast<::mlir::LLVM::LLVMStructType>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType()).isOpaque())))) && (!(((::llvm::isa<::mlir::LLVM::LLVMTargetExtType>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType()))) && ((!::llvm::cast<::mlir::LLVM::LLVMTargetExtType>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType()).supportsMemOps()))))) || ((::llvm::isa<::mlir::LLVM::PointerElementTypeInterface>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType())))" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.all_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %6 = irdl.c_pred "!::llvm::isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>($_self)" 
      %7 = irdl.all_of(%5, %6) 
      %8 = irdl.c_pred "!(((::llvm::isa<::mlir::LLVM::LLVMStructType>($_self))) && ((::llvm::cast<::mlir::LLVM::LLVMStructType>($_self).isOpaque())))" 
      %9 = irdl.c_pred "!(((::llvm::isa<::mlir::LLVM::LLVMTargetExtType>($_self))) && ((!::llvm::cast<::mlir::LLVM::LLVMTargetExtType>($_self).supportsMemOps())))" 
      %10 = irdl.all_of(%7, %8, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::LLVM::PointerElementTypeInterface>($_self)" 
      %12 = irdl.any_of(%10, %11) 
      irdl.operands(%4)
      irdl.results(%12)
    }
    irdl.operation @metadata {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @mul {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @mlir.null {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @or {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @mlir.poison {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @ptrtoint {
      %0 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @resume {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @return {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @sdiv {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @sext {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @sitofp {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @srem {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @select {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::getVectorElementType($_self).isSignlessInteger(1)" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands(%4, %5, %6)
      irdl.results(%7)
    }
    irdl.operation @shl {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @shufflevector {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @store {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      %1 = irdl.c_pred "!::llvm::isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>($_self)" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "!(((::llvm::isa<::mlir::LLVM::LLVMStructType>($_self))) && ((::llvm::cast<::mlir::LLVM::LLVMStructType>($_self).isOpaque())))" 
      %4 = irdl.c_pred "!(((::llvm::isa<::mlir::LLVM::LLVMTargetExtType>($_self))) && ((!::llvm::cast<::mlir::LLVM::LLVMTargetExtType>($_self).supportsMemOps())))" 
      %5 = irdl.all_of(%2, %3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::LLVM::PointerElementTypeInterface>($_self)" 
      %7 = irdl.any_of(%5, %6) 
      %8 = irdl.c_pred "::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)" 
      %9 = irdl.c_pred "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).isOpaque()" 
      %10 = irdl.c_pred "((((::mlir::LLVM::isCompatibleOuterType(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType()))) && ((!::llvm::isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType())))) && (!(((::llvm::isa<::mlir::LLVM::LLVMStructType>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType()))) && ((::llvm::cast<::mlir::LLVM::LLVMStructType>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType()).isOpaque())))) && (!(((::llvm::isa<::mlir::LLVM::LLVMTargetExtType>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType()))) && ((!::llvm::cast<::mlir::LLVM::LLVMTargetExtType>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType()).supportsMemOps()))))) || ((::llvm::isa<::mlir::LLVM::PointerElementTypeInterface>(::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getElementType())))" 
      %11 = irdl.any_of(%9, %10) 
      %12 = irdl.all_of(%8, %11) 
      irdl.operands(%7, %12)
      irdl.results()
    }
    irdl.operation @sub {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @switch {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.any
      %2 = irdl.any
      irdl.operands(%0, %1, %2)
      irdl.results()
    }
    irdl.operation @tbaa_root {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @tbaa_tag {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @tbaa_type_desc {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @trunc {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @udiv {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @uitofp {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
    irdl.operation @urem {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @mlir.undef {
      %0 = irdl.c_pred "::mlir::LLVM::isCompatibleOuterType($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @unreachable {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @xor {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      %10 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %11 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %12 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %13 = irdl.all_of(%11, %12) 
      %14 = irdl.any_of(%10, %13) 
      irdl.operands(%4, %9)
      irdl.results(%14)
    }
    irdl.operation @zext {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %6 = irdl.c_pred "::mlir::LLVM::isCompatibleVectorType($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>(::mlir::LLVM::getVectorElementType($_self))" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.any_of(%5, %8) 
      irdl.operands(%4)
      irdl.results(%9)
    }
  }
}
