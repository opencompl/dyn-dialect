// RUN: dyn-opt %s | dyn-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = "dyn.foo"(%{{.*}}) : (i32) -> i32
        %res1 = "dyn.foo"(%0) : (i32) -> i32
        // CHECK: %{{.*}} = "dyn.bar"(%{{.*}}) : (i32) -> i32
        %res2 = "dyn.bar"(%0) : (i32) -> i32

        return
    }
}