module {
  irdl.dialect @func {
    irdl.operation @call_indirect {
      %0 = irdl.c_pred "::llvm::isa<::mlir::FunctionType>($_self)" 
      %1 = irdl.any
      %2 = irdl.any
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @call {
      %0 = irdl.any
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @constant {
      %0 = irdl.any
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @func {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @return {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
  }
}
