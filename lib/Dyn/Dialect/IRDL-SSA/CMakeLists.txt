add_mlir_dialect_library(MLIRIRDLSSA
        IR/IRDLSSA.cpp
        IR/IRDLSSAAttributes.cpp
        IR/IRDLSSAOps.cpp
        IRDLSSARegistration.cpp
        IRDLSSAVerifiers.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dyn/Dialect/IRDL-SSA

        DEPENDS
        MLIRIRDLSSAIncGen
        MLIRIRDLSSAOpsIncGen
        MLIRIRDLSSATypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
