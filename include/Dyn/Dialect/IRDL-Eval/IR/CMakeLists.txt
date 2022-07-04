add_mlir_dialect(IRDLEval irdleval)

# Add IRDL-Eval operations
set(LLVM_TARGET_DEFINITIONS IRDLEvalOps.td)
mlir_tablegen(IRDLEvalOps.h.inc -gen-op-decls)
mlir_tablegen(IRDLEvalOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRIRDLEvalOpsIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLEvalOpsIncGen)

# Add IRDL-Eval types
set(LLVM_TARGET_DEFINITIONS IRDLEvalTypes.td)
mlir_tablegen(IRDLEvalTypesGen.h.inc -gen-typedef-decls) # TODO: Explain why this requires a different name
mlir_tablegen(IRDLEvalTypesGen.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRIRDLEvalTypesIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLEvalTypesIncGen)
