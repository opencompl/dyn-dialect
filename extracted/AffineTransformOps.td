module {
  irdl.dialect @transform {
    irdl.operation @affine.simplify_bounded_affine_ops {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0, %1)
      irdl.results()
    }
  }
}
