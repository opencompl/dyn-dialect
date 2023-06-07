module {
  irdl.dialect @complex {
    irdl.operation @abs {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @add {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      irdl.operands(%2, %5)
      irdl.results(%8)
    }
    irdl.operation @angle {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @atan2 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      irdl.operands(%2, %5)
      irdl.results(%8)
    }
    irdl.operation @conj {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @constant {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      irdl.operands()
      irdl.results(%2)
    }
    irdl.operation @cos {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @create {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      irdl.operands(%0, %1)
      irdl.results(%4)
    }
    irdl.operation @div {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      irdl.operands(%2, %5)
      irdl.results(%8)
    }
    irdl.operation @eq {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%2, %5)
      irdl.results(%6)
    }
    irdl.operation @exp {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @expm1 {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @im {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @log1p {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @log {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @mul {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      irdl.operands(%2, %5)
      irdl.results(%8)
    }
    irdl.operation @neg {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @neq {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%2, %5)
      irdl.results(%6)
    }
    irdl.operation @pow {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      irdl.operands(%2, %5)
      irdl.results(%8)
    }
    irdl.operation @re {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::FloatType>($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @rsqrt {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @sign {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @sin {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @sqrt {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @sub {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %8 = irdl.all_of(%6, %7) 
      irdl.operands(%2, %5)
      irdl.results(%8)
    }
    irdl.operation @tan {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @tanh {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::ComplexType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
  }
}
