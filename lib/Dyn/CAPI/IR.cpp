#include "Dyn/CAPI/IR.h"

using namespace mlir;
using namespace dyn;

// Context
MlirDynamicContext mlirDynamicContextCreate(MlirContext ctx) {
  DynamicContext *context = new DynamicContext(unwrap(ctx));
  return wrap(context);
}

void mlirDynamicContextDestroy(MlirDynamicContext context) {
  delete unwrap(context);
}

// Dialect
MlirStringRef mlirDynamicDialectGetNamespace(MlirDynamicDialect dialect) {
  return wrap(unwrap(dialect)->getName());
}

MlirDynamicDialect mlirDynamicDialectCreate(MlirStringRef prefix,
                                            MlirDynamicContext ctx) {
  DynamicDialect *dialect = new DynamicDialect(unwrap(prefix), unwrap(ctx));
  return wrap(dialect);
}

void mlirDynamicDialectDestroy(MlirDynamicDialect dialect) {
  delete unwrap(dialect);
}

// Operation
// TODO: When do we deallocate this operation?
// Option: when deallocating dynamicDialect (but it has no list atm).
// Option: mlirDynamicOperationDestroy (not good).
MlirDynamicOperation mlirDynamicOperationCreate(MlirStringRef name,
                                                MlirDynamicDialect dialect) {
  mlir::dyn::DynamicOperation *op =
      new DynamicOperation(unwrap(name), unwrap(dialect), {}, {}, {});
  return wrap(op);
}

void mlirDynamicOperationDestroy(MlirDynamicOperation operation) {
  delete unwrap(operation);
}

MlirStringRef mlirDynamicOperationGetName(MlirDynamicOperation operation) {
  return wrap(unwrap(operation)->getName());
}
