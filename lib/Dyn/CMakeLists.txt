add_subdirectory(CAPI)
add_subdirectory(Dialect)
add_mlir_dialect_library(MLIRDyn
        DynamicObject.cpp
        DynamicDialect.cpp
        DynamicContext.cpp
        DynamicType.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dyn

	LINK_LIBS PUBLIC
	MLIRIR
	)
