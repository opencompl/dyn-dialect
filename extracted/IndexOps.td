module {
  irdl.dialect @index {
    irdl.operation @add {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @and {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @bool.constant {
      %0 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @casts {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.any_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @castu {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.any_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::IntegerType>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %5 = irdl.any_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @ceildivs {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @ceildivu {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @cmp {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "$_self.isSignlessInteger(1)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @constant {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @divs {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @divu {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @floordivs {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @maxs {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @maxu {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @mins {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @minu {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @mul {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @or {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @rems {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @remu {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @shl {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @shrs {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @shru {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @sizeof {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @sub {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @xor {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::IndexType>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
  }
}
