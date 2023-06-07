module {
  irdl.dialect @cf {
    irdl.operation @assert {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @br {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @cond_br {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.any
      %2 = irdl.any
      irdl.operands(%0, %1, %2)
      irdl.results()
    }
    irdl.operation @switch {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.any
      %2 = irdl.any
      irdl.operands(%0, %1, %2)
      irdl.results()
    }
  }
}
