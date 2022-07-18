add_mlir_dialect_library(MLIRIRDLEval
        IR/IRDLEval.cpp
        IR/IRDLEvalOps.cpp
        IRDLEvalInterpreter.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dyn/Dialect/IRDL-Eval

  LINK_LIBS PUBLIC
  MLIRIR
  )
