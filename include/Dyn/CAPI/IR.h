#ifndef DYN_CAPI_IR_H
#define DYN_CAPI_IR_H

#include "Dyn-c/IR.h"
#include "Dyn/DynamicContext.h"
#include "Dyn/DynamicDialect.h"
#include "Dyn/DynamicOperation.h"
#include "mlir/CAPI/Wrap.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

DEFINE_C_API_PTR_METHODS(MlirDynamicContext, mlir::dyn::DynamicContext)
DEFINE_C_API_PTR_METHODS(MlirDynamicDialect, mlir::dyn::DynamicDialect)
DEFINE_C_API_PTR_METHODS(MlirDynamicOperation, mlir::dyn::DynamicOperation)

#endif // DYN_CAPI_IR_H
