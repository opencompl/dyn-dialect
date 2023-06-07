module {
  irdl.dialect @transform {
    irdl.operation @apply_patterns.linalg.erase_unnecessary_inputs {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @apply_patterns.linalg.fold_unit_extent_dims_via_reshapes {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @apply_patterns.linalg.fold_unit_extent_dims_via_slices {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @apply_patterns.linalg.tiling_canonicalization {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @structured.bufferize_to_allocation {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::AnyValueType>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::AnyValueType>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @structured.convert_conv2d_to_img2col {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1, %2)
    }
    irdl.operation @structured.decompose {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @structured.fuse_into_containing_op {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2, %3)
    }
    irdl.operation @structured.fuse {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1, %2)
    }
    irdl.operation @structured.generalize {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @structured.hoist_pad.build_packing_loop_nest {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @structured.hoist_pad {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @structured.hoist_redundant_tensor_subsets {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @structured.hoist_redundant_vector_transfers {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @structured.insert_slice_to_copy {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @structured.interchange {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @structured.lower_pack {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22tensor.pack\22" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22tensor.pad\22" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22tensor.expand_shape\22" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22linalg.transpose\22" 
      %11 = irdl.all_of(%9, %10) 
      irdl.operands(%2)
      irdl.results(%5, %8, %11)
    }
    irdl.operation @structured.lower_unpack {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %1 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22tensor.unpack\22" 
      %2 = irdl.all_of(%0, %1) 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %4 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22tensor.empty\22" 
      %5 = irdl.all_of(%3, %4) 
      %6 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %7 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22linalg.transpose\22" 
      %8 = irdl.all_of(%6, %7) 
      %9 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %10 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22tensor.collapse_shape\22" 
      %11 = irdl.all_of(%9, %10) 
      %12 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %13 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22tensor.extract_slice\22" 
      %14 = irdl.all_of(%12, %13) 
      irdl.operands(%2)
      irdl.results(%5, %8, %11, %14)
    }
    irdl.operation @structured.masked_vectorize {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0, %1)
      irdl.results()
    }
    irdl.operation @structured.match {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @structured.multitile_sizes {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::ParamType>($_self)" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::transform::ParamType>($_self)" 
      %6 = irdl.any_of(%4, %5) 
      %7 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %8 = irdl.c_pred "::llvm::isa<::mlir::transform::ParamType>($_self)" 
      %9 = irdl.any_of(%7, %8) 
      irdl.operands(%0)
      irdl.results(%3, %6, %9)
    }
    irdl.operation @structured.pack_greedily {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::OperationType>($_self)" 
      %3 = irdl.c_pred "::llvm::cast<::mlir::transform::OperationType>($_self).getOperationName() == \22linalg.generic\22" 
      %4 = irdl.all_of(%2, %3) 
      irdl.operands(%0, %1)
      irdl.results(%4)
    }
    irdl.operation @structured.pack {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2)
    }
    irdl.operation @structured.pack_transpose {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2, %3, %4)
    }
    irdl.operation @structured.pad {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @structured.promote {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @structured.replace {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @structured.rewrite_in_destination_passing_style {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @structured.scalarize {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @structured.split {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::ParamType>($_self)" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0, %3)
      irdl.results(%4, %5)
    }
    irdl.operation @structured.split_reduction {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1, %2, %3, %4)
    }
    irdl.operation @structured.tile {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::ParamType>($_self)" 
      %3 = irdl.any_of(%1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0, %3)
      irdl.results(%4, %5)
    }
    irdl.operation @structured.tile_reduction_using_forall {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1, %2, %3, %4)
    }
    irdl.operation @structured.tile_reduction_using_scf {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1, %2, %3, %4)
    }
    irdl.operation @structured.tile_to_forall_op {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %4 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0, %1, %2, %3, %4)
      irdl.results(%5, %6)
    }
    irdl.operation @structured.tile_to_scf_for {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0, %1)
      irdl.results(%2, %3)
    }
    irdl.operation @structured.vectorize {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
  }
}
