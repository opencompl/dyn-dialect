module {
  irdl.dialect @irdl {
    irdl.operation @all_of {
      %0 = irdl.c_pred "::llvm::isa<::mlir::irdl::AttributeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::irdl::AttributeType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @any_of {
      %0 = irdl.c_pred "::llvm::isa<::mlir::irdl::AttributeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::irdl::AttributeType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @any {
      %0 = irdl.c_pred "::llvm::isa<::mlir::irdl::AttributeType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @attribute {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @c_pred {
      %0 = irdl.c_pred "::llvm::isa<::mlir::irdl::AttributeType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @dialect {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @is {
      %0 = irdl.c_pred "::llvm::isa<::mlir::irdl::AttributeType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @operands {
      %0 = irdl.c_pred "::llvm::isa<::mlir::irdl::AttributeType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @operation {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @parameters {
      %0 = irdl.c_pred "::llvm::isa<::mlir::irdl::AttributeType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @parametric {
      %0 = irdl.c_pred "::llvm::isa<::mlir::irdl::AttributeType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::irdl::AttributeType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @results {
      %0 = irdl.c_pred "::llvm::isa<::mlir::irdl::AttributeType>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @type {
      irdl.operands()
      irdl.results()
    }
  }
}
