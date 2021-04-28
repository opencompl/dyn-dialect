add_mlir_dialect(IRDLOps irdl)

# Add IRDL attributes
set(LLVM_TARGET_DEFINITIONS IRDLInterfaces.td)
mlir_tablegen(IRDLInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(IRDLInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRIRDLInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLInterfacesIncGen)

add_mlir_doc(IRDLOps -gen-dialect-doc IRDLOps Dialects/)
