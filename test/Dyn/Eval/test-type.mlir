// RUN: dyn-opt %s --irdlssa-file=%S/test-type.irdlssa -split-input-file -verify-diagnostics | FileCheck %s

func.func @succeededTypeVerifier() {
    // CHECK: "testd.any"() : () -> !testd.singleton
    "testd.any"() : () -> !testd.singleton

    // CHECK-NEXT: "testd.any"() : () -> !testd.parametrized<f32, i32>
    "testd.any"() : () -> !testd.parametrized<f32, i32>

    // CHECK: "testd.any"() : () -> !testd.parametrized<i1, i64>
    "testd.any"() : () -> !testd.parametrized<i1, i64>

    return
}

// -----

func.func @failedSingletonVerifier() {
     // expected-error@+1 {{invalid amount of types, expected 0, got 1}}
     "testd.any"() : () -> !testd.singleton<i32>
}

// -----

func.func @failedParametrizedVerifierWrongNumOfArgs() {
     // expected-error@+1 {{invalid amount of types, expected 2, got 1}}
     "testd.any"() : () -> !testd.parametrized<i32>
}

// -----

func.func @failedParametrizedVerifierWrongArgument() {
     // expected-error@+1 {{the provided types do not satisfy the type constraints}}
     "testd.any"() : () -> !testd.parametrized<i32, i1>
}
