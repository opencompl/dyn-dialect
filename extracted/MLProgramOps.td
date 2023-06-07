module {
  irdl.dialect @ml_program {
    irdl.operation @func {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @global_load_const {
      %0 = irdl.any
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @global_load_graph {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ml_program::TokenType>($_self)" 
      %1 = irdl.any
      %2 = irdl.c_pred "::llvm::isa<::mlir::ml_program::TokenType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1, %2)
    }
    irdl.operation @global_load {
      %0 = irdl.any
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @global {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @global_store_graph {
      %0 = irdl.any
      %1 = irdl.c_pred "::llvm::isa<::mlir::ml_program::TokenType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::ml_program::TokenType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @global_store {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @output {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @return {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @subgraph {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @token {
      %0 = irdl.c_pred "::llvm::isa<::mlir::ml_program::TokenType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
  }
}
