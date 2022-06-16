add_mlir_dialect_library(MLIRIRDL
        IR/IRDL.cpp
        IR/IRDLAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dyn/Dialect/IRDL

        DEPENDS
        MLIRIRDLIncGen
        MLIRIRDLOpsIncGen
        MLIRIRDLInterfacesIncGen
        MLIRIRDLAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

