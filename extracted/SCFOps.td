module {
  irdl.dialect @scf {
    irdl.operation @condition {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.any
      irdl.operands(%0, %1)
      irdl.results()
    }
    irdl.operation @execute_region {
      %0 = irdl.any
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @for {
      %0 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %1 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %2 = irdl.c_pred "$_self.isSignlessIntOrIndex()" 
      %3 = irdl.any
      %4 = irdl.any
      irdl.operands(%0, %1, %2, %3)
      irdl.results(%4)
    }
    irdl.operation @forall {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::RankedTensorType>($_self)" 
      %4 = irdl.all_of(%3) 
      %5 = irdl.c_pred "[](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())" 
      %6 = irdl.all_of(%4, %5) 
      %7 = irdl.any
      irdl.operands(%0, %1, %2, %6)
      irdl.results(%7)
    }
    irdl.operation @if {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @forall.in_parallel {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @index_switch {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @parallel {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %3 = irdl.any
      %4 = irdl.any
      irdl.operands(%0, %1, %2, %3)
      irdl.results(%4)
    }
    irdl.operation @reduce {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @reduce.return {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @while {
      %0 = irdl.any
      %1 = irdl.any
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @yield {
      %0 = irdl.any
      irdl.operands(%0)
      irdl.results()
    }
  }
}
