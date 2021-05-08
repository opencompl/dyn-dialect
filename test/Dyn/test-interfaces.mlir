// RUN: dyn-opt %s --irdl-file=%S/cmath.irdl -canonicalize | FileCheck %s

module {
    // CHECK-LABEL: func @bar(%{{.*}}: !cmath.real, %{{.*}}: !cmath.real) {
    func @bar(%re: !cmath.real, %im: !cmath.real) {
        %c = "cmath.make_complex"(%re, %im) : (!cmath.real, !cmath.real) -> !cmath.ccomplex
        // CHECK-NEXT: return
        return
    }
}
