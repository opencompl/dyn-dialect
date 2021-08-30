set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(dialect-stats
  dialect-stats.cpp
  )

llvm_update_compile_flags(dialect-stats)
target_link_libraries(dialect-stats
  PRIVATE
  ${dialect_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
  MLIRTableGen
  )

mlir_check_link_libraries(dialect-stats)
