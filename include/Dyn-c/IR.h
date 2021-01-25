#ifndef DYN_C_IR_H
#define DYN_C_IR_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirDynamicContext, void);
DEFINE_C_API_STRUCT(MlirDynamicDialect, void);
DEFINE_C_API_STRUCT(MlirDynamicOperation, void);

#undef DEFINE_C_API_STRUCT

MLIR_CAPI_EXPORTED MlirDynamicContext mlirDynamicContextCreate(MlirContext ctx);
MLIR_CAPI_EXPORTED void mlirDynamicContextDestroy(MlirDynamicContext ctx);

MLIR_CAPI_EXPORTED MlirStringRef
mlirDynamicDialectGetNamespace(MlirDynamicDialect dialect);
MLIR_CAPI_EXPORTED MlirDynamicDialect
mlirDynamicDialectCreate(MlirStringRef prefix, MlirDynamicContext ctx);
MLIR_CAPI_EXPORTED void mlirDynamicDialectDestroy(MlirDynamicDialect dialect);

MLIR_CAPI_EXPORTED MlirDynamicOperation
mlirDynamicOperationCreate(MlirStringRef name, MlirDynamicDialect ctx);
MLIR_CAPI_EXPORTED MlirStringRef
mlirDynamicOperationGetName(MlirDynamicOperation op);
MLIR_CAPI_EXPORTED void mlirDynamicOperationDestroy(MlirDynamicOperation op);

#ifdef __cplusplus
}
#endif

#endif // DYN_CAPI_IR_H
