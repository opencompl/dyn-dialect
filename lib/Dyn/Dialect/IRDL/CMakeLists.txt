add_mlir_dialect_library(MLIRIRDL
        IR/IRDL.cpp
        IR/IRDLAttributes.cpp
        IR/StandardOpInterfaces.cpp
        IRDLRegistration.cpp
        TypeConstraint.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dyn/Dialect/IRDL

        DEPENDS
        MLIRIRDLOpsIncGen
        MLIRIRDLInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

