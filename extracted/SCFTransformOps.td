module {
  irdl.dialect @transform {
    irdl.operation @apply_patterns.scf.for_loop_canonicalization {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @loop.get_parent_for {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @loop.coalesce {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @loop.outline {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1, %2)
    }
    irdl.operation @loop.peel {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22scf.for\22" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @loop.pipeline {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22scf.for\22" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
    irdl.operation @loop.unroll {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @scf.take_assumed_branch {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
  }
}
