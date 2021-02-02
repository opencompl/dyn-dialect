// RUN: dyn-opt %s --irdl-file=%S/cmath.irdl" | FileCheck %s

module {
    // CHECK-LABEL: func @bar() {
    func @bar() {
        // CHECK: "cmath.dummyop"() : () -> ()
        "cmath.dummyop"() : () -> ()
       return
    }
}