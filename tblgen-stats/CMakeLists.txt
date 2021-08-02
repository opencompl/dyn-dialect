set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(tblgen-stats
  tblgen-stats.cpp
  )

llvm_update_compile_flags(tblgen-stats)
target_link_libraries(tblgen-stats
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

mlir_check_link_libraries(tblgen-stats)
