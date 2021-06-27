// RUN: dyn-opt %s -split-input-file -verify-diagnostics | dyn-opt | FileCheck %s

// CHECK-LABEL: irdl.dialect testd {

irdl.dialect testd {
    // CHECK: irdl.type parametric<param: irdl.Any>
    irdl.type parametric<param: irdl.Any>

    // CHECK: irdl.operation eq() -> (res: i32)
    irdl.operation eq() -> (res: i32)
    // CHECK: irdl.operation anyof() -> (res: irdl.AnyOf<i32, i64>)
    irdl.operation anyof() -> (res: irdl.AnyOf<i32, i64>)

    // CHECK: irdl.operation any() -> (res: irdl.Any)
    irdl.operation any() -> (res: irdl.Any)

    // CHECK: irdl.operation dynparams() -> (res: testd.parametric<irdl.AnyOf<i32, i64>>)
    irdl.operation dynparams() -> (res: testd.parametric<irdl.AnyOf<i32, i64>>)
}

// -----

//===----------------------------------------------------------------------===//
// Equality constraint
//===----------------------------------------------------------------------===//

func @succeededEqConstraint() {
    // CHECK: "testd.eq"() : () -> i32
    "testd.eq"() : () -> i32
    return
}

// -----

func @failedEqConstraint() {
    // expected-error@+1 {{expected type 'i32' but got type 'i64'}}
    "testd.eq"() : () -> i64
    return
}

// -----

//===----------------------------------------------------------------------===//
// AnyOf constraint
//===----------------------------------------------------------------------===//

func @succeededAnyOfConstraint() {
    // CHECK: "testd.anyof"() : () -> i32
    "testd.anyof"() : () -> i32
    // CHECK: "testd.anyof"() : () -> i64
    "testd.anyof"() : () -> i64
    return
}

// -----

func @failedAnyOfConstraint() {
    // expected-error@+1 {{type 'i1' does not satisfy the constraint}}
    "testd.anyof"() : () -> i1
    return
}

// -----

//===----------------------------------------------------------------------===//
// Any constraint
//===----------------------------------------------------------------------===//

func @succeededAnyConstraint() {
    // CHECK: "testd.any"() : () -> i32
    "testd.any"() : () -> i32
    // CHECK: "testd.any"() : () -> i64
    "testd.any"() : () -> i64
    return
}

// -----

//===----------------------------------------------------------------------===//
// Dynamic parameters constraint
//===----------------------------------------------------------------------===//

func @succeededDynParamsConstraint() {
    // CHECK: "testd.dynparams"() : () -> !testd.parametric<i32>
    "testd.dynparams"() : () -> !testd.parametric<i32>
    // CHECK: "testd.dynparams"() : () -> !testd.parametric<i64>
    "testd.dynparams"() : () -> !testd.parametric<i64>
    return
}

// -----

func @failedDynParamsConstraint() {
    // expected-error@+1 {{type 'i1' does not satisfy the constraint}}
    "testd.dynparams"() : () -> !testd.parametric<i1>
    return
}