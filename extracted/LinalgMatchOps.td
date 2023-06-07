module {
  irdl.dialect @transform {
    irdl.operation @match.structured.body {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results()
    }
    irdl.operation @match.structured.dim {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @match.structured.elemental_bitwidth {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @match.structured.init {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      %3 = irdl.any_of(%1, %2) 
      irdl.operands(%0)
      irdl.results(%3)
    }
    irdl.operation @match.structured.input {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      %3 = irdl.any_of(%1, %2) 
      irdl.operands(%0)
      irdl.results(%3)
    }
    irdl.operation @match.structured.num_inits {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @match.structured.num_inputs {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @match.structured {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %3 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      %4 = irdl.any_of(%1, %2, %3) 
      irdl.operands(%0)
      irdl.results(%4)
    }
    irdl.operation @match.structured.rank {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      irdl.operands(%0)
      irdl.results(%1)
    }
    irdl.operation @match.structured.result {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      %3 = irdl.any_of(%1, %2) 
      irdl.operands(%0)
      irdl.results(%3)
    }
    irdl.operation @match.structured.yield {
      %0 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformParamTypeInterface>($_self)" 
      %1 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformHandleTypeInterface>($_self)" 
      %2 = irdl.c_pred "::llvm::isa<::mlir::transform::TransformValueHandleTypeInterface>($_self)" 
      %3 = irdl.any_of(%0, %1, %2) 
      irdl.operands(%3)
      irdl.results()
    }
  }
}
