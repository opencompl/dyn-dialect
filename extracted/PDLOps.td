module {
  irdl.dialect @pdl {
    irdl.operation @apply_native_constraint {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @apply_native_rewrite {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @attribute {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::AttributeType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @erase {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @operand {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @operands {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @operation {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %3 = irdl.all_of(%1, %2) 
      %4 = irdl.any_of(%0, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::pdl::AttributeType>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %8 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %9 = irdl.all_of(%7, %8) 
      %10 = irdl.any_of(%6, %9) 
      %11 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      irdl.operands(%4, %5, %10)
      irdl.results(%11)
    }
    irdl.operation @pattern {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @range {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %2 = irdl.c_pred "((::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType()))) || ((::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())))" 
      %3 = irdl.all_of(%1, %2) 
      irdl.operands(%0)
      irdl.results(%3)
    }
    irdl.operation @replace {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.any_of(%2, %5) 
      irdl.operands(%0, %1, %6)
      irdl.results()
    }
    irdl.operation @result {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @results {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %4 = irdl.all_of(%2, %3) 
      %5 = irdl.any_of(%1, %4) 
      irdl.operands(%0)
      irdl.results(%5)
    }
    irdl.operation @rewrite {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::PDLType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results()
    }
    irdl.operation @type {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @types {
      %0 = irdl.c_pred "::llvm::isa<::mlir::pdl::RangeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>($_self).getElementType())" 
      %2 = irdl.all_of(%0, %1) 
      irdl.operands()
      irdl.results(%2)
    }
  }
}
