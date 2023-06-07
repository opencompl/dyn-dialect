module {
  irdl.dialect @transform {
    irdl.operation @alternatives {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @annotate {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      irdl.operands(%0, %1)
      irdl.results()
    }
    irdl.operation @apply_patterns.canonicalization {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @apply_patterns {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @cast {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @foreach_match {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @foreach {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @get_closest_isolated_parent {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @get_consumers_of_result {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @get_defining_op {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @get_producer_of_operand {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @get_result {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @include {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      %3 = irdl.any_of(%0, %1, %2) 
      %4 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      %5 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      %7 = irdl.any_of(%4, %5, %6) 
      irdl.operands(%3)
      irdl.results(%7)
    }
    irdl.operation @match.operation_name {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @match.param.cmpi {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      irdl.operands(%0, %1)
      irdl.results()
    }
    irdl.operation @merge_handles {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @named_sequence {
      irdl.operands()
      irdl.results()
    }
    irdl.operation @param.constant {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      irdl.operands()
      irdl.results(%0)
    }
    irdl.operation @print {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @replicate {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      %4 = irdl.any_of(%1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      %6 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %7 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      %8 = irdl.any_of(%5, %6, %7) 
      irdl.operands(%0, %4)
      irdl.results(%8)
    }
    irdl.operation @sequence {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      %4 = irdl.any_of(%1, %2, %3) 
      %5 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0, %4)
      irdl.results(%5)
    }
    irdl.operation @split_handle {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @yield {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      %3 = irdl.any_of(%0, %1, %2) 
      irdl.operands(%3)
      irdl.results()
    }
  }
}
