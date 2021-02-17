add_mlir_dialect(IRDLOps irdl)

# Add IRDL attributes
set(LLVM_TARGET_DEFINITIONS IRDLInterface.td)
mlir_tablegen(IRDLInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(IRDLInterface.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRIRDLInterfaceIncGen)
add_dependencies(mlir-generic-headers MLIRIRDLInterfaceIncGen)

add_mlir_doc(IRDLOps -gen-dialect-doc IRDLOps Dialects/)
