module {
  irdl.dialect @linalg {
    irdl.operation @index {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @yield {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
  }
}
