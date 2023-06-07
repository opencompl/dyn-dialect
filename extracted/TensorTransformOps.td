module {
  irdl.dialect @transform {
    irdl.operation @apply_patterns.tensor.drop_redundant_insert_slice_rank_expansion {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @apply_patterns.tensor.fold_into_pack_and_unpack {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @apply_patterns.tensor.fold_tensor_empty {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @apply_patterns.tensor.fold_tensor_subset_ops {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @apply_patterns.tensor.merge_consecutive_insert_extract_slice {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @apply_patterns.tensor.reassociative_reshape_folding {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @tensor.make_loop_independent {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
  }
}
