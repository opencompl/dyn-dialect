module {
  irdl.dialect @transform {
    irdl.operation @apply_patterns.memref.expand_ops {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @apply_patterns.memref.expand_strided_metadata {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @apply_patterns.memref.extract_address_computations {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @apply_patterns.memref.fold_memref_alias_ops {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @apply_patterns.memref.resolve_ranked_shaped_type_result_dims {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @memref.make_loop_independent {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @memref.multibuffer {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22memref.alloc\22" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%2)
      irdl.results(%3)
    }
  }
}
