// RUN: dyn-opt %s --irdl-file=%S/cmath.irdl | dyn-opt --irdl-file=%S/cmath.irdl | FileCheck %s

module {
    // CHECK-LABEL: func @bar(%{{.*}}: !cmath.real, %{{.*}}: !cmath.real) {
    func @bar(%re: !cmath.real, %im: !cmath.real) {
        // CHECK: %{{.*}} = "cmath.make_complex"(%{{.*}}, %{{.*}}) : (!cmath.real, !cmath.real) -> !cmath.complex
        %c = "cmath.make_complex"(%re, %im) : (!cmath.real, !cmath.real) -> !cmath.complex
        // CHECK: %{{.*}} = "cmath.mul"(%{{.*}}, %{{.*}}) : (!cmath.complex, !cmath.complex) -> !cmath.complex
        %c2 = "cmath.mul"(%c, %c) : (!cmath.complex, !cmath.complex) -> !cmath.complex
        // CHECK: %{{.*}} = "cmath.norm"(%{{.*}}) : (!cmath.complex) -> !cmath.real
        %resnorm = "cmath.norm"(%c2) : (!cmath.complex) -> !cmath.real
        // CHECK: %{{.*}} = "cmath.get_real"(%{{.*}}) : (!cmath.complex) -> !cmath.real
        %res_real = "cmath.get_real"(%c2) : (!cmath.complex) -> !cmath.real
        // CHECK: %{{.*}} = "cmath.get_imaginary"(%{{.*}}) : (!cmath.complex) -> !cmath.real
        %res_im = "cmath.get_imaginary"(%c2) : (!cmath.complex) -> !cmath.real
        return
    }
}