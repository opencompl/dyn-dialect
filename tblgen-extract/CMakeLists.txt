set(LLVM_LINK_COMPONENTS
  TableGen
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        )
add_llvm_executable(tblgen-extract tblgen-extract.cpp)

llvm_update_compile_flags(tblgen-extract)
target_link_libraries(tblgen-extract
  PRIVATE
  ${LIBS}
  MLIRIR
  MLIRTableGen
  MLIRSupport
  )

mlir_check_all_link_libraries(tblgen-extract)
