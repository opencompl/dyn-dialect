add_mlir_dialect_library(MLIRIRDL
        IRDL.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dyn/Dialect/IRDL/IR

        DEPENDS
        MLIRIRDLOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
