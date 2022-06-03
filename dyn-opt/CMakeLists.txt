get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        )
add_llvm_executable(dyn-opt dyn-opt.cpp MlirOptMain.cpp RegisterIRDL.cpp LowerIRDL.cpp)

llvm_update_compile_flags(dyn-opt)
target_link_libraries(dyn-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(dyn-opt)
