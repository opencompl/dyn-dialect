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
  return wrap(unwrap(dialect)->getNamespace());
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
  // auto res =
  // unwrap(dialect)->getDynamicContext()->createAndRegisterOperation(
  //    unwrap(name), unwrap(dialect), {}, {}, {});
  // TODO rewrite this when the MLIR API is fixed.
  // assert(
  //     succeeded(res) &&
  //     "Trying to register a dynamic operation with an already existing
  //     name.");
  // return wrap(*res);
  assert(false && "Can't create new dynamic operations");
}

void mlirDynamicOperationDestroy(MlirDynamicOperation operation) {}

MlirStringRef mlirDynamicOperationGetName(MlirDynamicOperation operation) {
  assert(false && "Can't get the name of a dynamic operation");
  // return wrap(unwrap(operation)->getName());
}
