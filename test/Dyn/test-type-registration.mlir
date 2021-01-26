// RUN: dyn-opt %s | dyn-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        // CHECK: %{{.*}} = "dyn.foo"() : () -> !dyn.dyntype
        %res = "dyn.foo"() : () -> !dyn.dyntype

        return
    }
}