add_mlir_dialect_library(MLIRDyn
        DynDialect.cpp
        DynOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dyn

        DEPENDS
        MLIRDynOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
