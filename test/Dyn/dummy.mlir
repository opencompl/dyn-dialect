// RUN: dyn-opt %s | dyn-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = "dyn.dynop"(%{{.*}}) : (i32) -> i32
        %res = "dyn.dynop"(%0) : (i32) -> i32
        return
    }
}
