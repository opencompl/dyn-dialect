add_mlir_dialect(IRDL irdl)

# Add IRDL interfaces
set(LLVM_TARGET_DEFINITIONS IRDLInterfaces.td)
mlir_tablegen(IRDLInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(IRDLInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRIRDLInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLInterfacesIncGen)

# Add IRDL attributes
set(LLVM_TARGET_DEFINITIONS IRDLAttributes.td)
mlir_tablegen(IRDLAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(IRDLAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRIRDLAttributesIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLAttributesIncGen)

# Add IRDL standard OpInterface attributes
set(LLVM_TARGET_DEFINITIONS StandardOpInterfaces.td)
mlir_tablegen(StandardOpInterfaces.h.inc -gen-attrdef-decls)
mlir_tablegen(StandardOpInterfaces.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRStandardOpInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIRStandardOpInterfacesIncGen)

# Add IRDL operations
set(LLVM_TARGET_DEFINITIONS IRDLOps.td)
mlir_tablegen(IRDLOps.h.inc -gen-op-decls)
mlir_tablegen(IRDLOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRIRDLOpsIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLOpsIncGen)
