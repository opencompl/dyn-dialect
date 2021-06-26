// RUN: dyn-opt %s --irdl-file=%S/test-type.irdl -split-input-file -verify-diagnostics | FileCheck %s

func @succeededTypeVerifier() {
    // CHECK: "testd.any"() : () -> !testd.singleton
    "testd.any"() : () -> !testd.singleton

    // CHECK-NEXT: "testd.any"() : () -> !testd.parametrized<f32, i32>
    "testd.any"() : () -> !testd.parametrized<f32, i32>

    // CHECK: "testd.any"() : () -> !testd.parametrized<i1, i64>
    "testd.any"() : () -> !testd.parametrized<i1, i64>

    return
}

// -----

func @failedSingletonVerifier() {
     // expected-error@+1 {{expected 0 type arguments, but had 1}}
     "testd.any"() : () -> !testd.singleton<i32>
}

// -----

func @failedParametrizedVerifierWrongNumOfArgs() {
     // expected-error@+1 {{expected 2 type arguments, but had 1}}
     "testd.any"() : () -> !testd.parametrized<i32>
}

// -----

func @failedParametrizedVerifierWrongArgument() {
     // expected-error@+1 {{invalid parameter type}}
     "testd.any"() : () -> !testd.parametrized<i32, i1>
}