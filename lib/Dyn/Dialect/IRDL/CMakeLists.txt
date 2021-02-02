add_mlir_dialect_library(MLIRIRDL
        IR/IRDL.cpp
        IRDLRegistration.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dyn/Dialect/IRDL

        DEPENDS
        MLIRIRDLOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

