add_mlir_dialect_library(MLIRIRDL
        IR/IRDL.cpp
        IR/IRDLAttributes.cpp
        IRDLRegistration.cpp
        TypeConstraint.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dyn/Dialect/IRDL

        DEPENDS
        MLIRIRDLOpsIncGen
        MLIRIRDLInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

