// RUN: mlir-dyn-capi-pass-test | FileCheck %s

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Dyn-c/IR.h"
#include "mlir-c/IR.h"

void testModule() {
  MlirContext ctx = mlirContextCreate();
  MlirDynamicContext dynCtx = mlirDynamicContextCreate(ctx);
  MlirDynamicDialect dynDialect =
      mlirDynamicDialectCreate(mlirStringRefCreateFromCString("dyn"), dynCtx);
  MlirStringRef prefix = mlirDynamicDialectGetNamespace(dynDialect);

  // CHECK: dyn
  fwrite(prefix.data, prefix.length, 1, stdout);
  fwrite("\n", sizeof(char), 1, stdout);

  MlirDynamicOperation operation = 
    mlirDynamicOperationCreate(mlirStringRefCreateFromCString("foo"), dynDialect);
  prefix = mlirDynamicOperationGetName(operation);

  // CHECK: dyn.foo
  fwrite(prefix.data, prefix.length, 1, stdout);
  fwrite("\n", sizeof(char), 1, stdout);

  mlirDynamicOperationDestroy(operation);
  mlirDynamicDialectDestroy(dynDialect);
  mlirDynamicContextDestroy(dynCtx);
  mlirContextDestroy(ctx);
}

int main() {
  testModule();
  return 0;
}
