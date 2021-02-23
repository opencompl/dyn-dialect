// RUN: dyn-opt %s --irdl-file=%S/cmath.irdl | dyn-opt --irdl-file=%S/cmath.irdl | FileCheck %s

module {
    // CHECK-LABEL: func @bar(%{{.*}}: f32, %{{.*}}: f32) {
    func @bar(%re: f32, %im: f32) {
        // CHECK: %{{.*}} = "cmath.make_complex"(%{{.*}}, %{{.*}}) : (f32, f32) -> !cmath.complex
        %c = "cmath.make_complex"(%re, %im) : (f32, f32) -> !cmath.complex
        // CHECK: %{{.*}} = "cmath.mul"(%{{.*}}, %{{.*}}) : (!cmath.complex, !cmath.complex) -> !cmath.complex
        %c2 = "cmath.mul"(%c, %c) : (!cmath.complex, !cmath.complex) -> !cmath.complex
        // CHECK: %{{.*}} = "cmath.norm"(%{{.*}}) : (!cmath.complex) -> f32
        %resnorm = "cmath.norm"(%c2) : (!cmath.complex) -> f32
        // CHECK: %{{.*}} = "cmath.get_real"(%{{.*}}) : (!cmath.complex) -> f32
        %res_real = "cmath.get_real"(%c2) : (!cmath.complex) -> f32
        // CHECK: %{{.*}} = "cmath.get_imaginary"(%{{.*}}) : (!cmath.complex) -> f32
        %res_im = "cmath.get_imaginary"(%c2) : (!cmath.complex) -> f32
        return
    }
}