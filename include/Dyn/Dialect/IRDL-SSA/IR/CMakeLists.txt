add_mlir_dialect(IRDLSSA irdlssa)

# Add IRDL-SSA interfaces
set(LLVM_TARGET_DEFINITIONS IRDLSSAInterfaces.td)
mlir_tablegen(IRDLSSAInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(IRDLSSAInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRIRDLSSAInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLSSAInterfacesIncGen)

# Add IRDL-SSA operations
set(LLVM_TARGET_DEFINITIONS IRDLSSAOps.td)
mlir_tablegen(IRDLSSAOps.h.inc -gen-op-decls)
mlir_tablegen(IRDLSSAOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRIRDLSSAOpsIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLSSAOpsIncGen)

# Add IRDL-SSA types
set(LLVM_TARGET_DEFINITIONS IRDLSSATypes.td)
mlir_tablegen(IRDLSSATypesGen.h.inc -gen-typedef-decls) # TODO: Explain why this requires a different name
mlir_tablegen(IRDLSSATypesGen.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRIRDLSSATypesIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLSSATypesIncGen)

# Add IRDL-SSA attributes
set(LLVM_TARGET_DEFINITIONS IRDLSSAAttributes.td)
mlir_tablegen(IRDLSSAAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(IRDLSSAAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRIRDLSSAAttributesIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLSSAAttributesIncGen)
