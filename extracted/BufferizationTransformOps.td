module {
  irdl.dialect @transform {
    irdl.operation @bufferization.eliminate_empty_tensors {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @bufferization.empty_tensor_to_alloc_tensor {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22tensor.empty\22" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22bufferization.alloc_tensor\22" 
      %5 = irdl.all_of(%3, %4) 
      irdl.operands(%2)
      irdl.results(%5)
    }
    irdl.operation @bufferization.one_shot_bufferize {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
  }
}
