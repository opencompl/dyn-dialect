// RUN: dyn-opt %s --irdl-file=%S/cmath.irdl | dyn-opt --irdl-file=%S/cmath.irdl | FileCheck %s

module {
    // CHECK-LABEL: func @bar(%{{.*}}: f32, %{{.*}}: f32) {
    func @bar(%re: f32, %im: f32) {
        // CHECK: %{{.*}} = "cmath.make_complex"(%{{.*}}, %{{.*}}) : (f32, f32) -> !cmath.ccomplex<f32>
        %c = "cmath.make_complex"(%re, %im) : (f32, f32) -> !cmath.ccomplex<f32>
        // CHECK: %{{.*}} = "cmath.mul"(%{{.*}}, %{{.*}}) : (!cmath.ccomplex<f32>, !cmath.ccomplex<f32>) -> !cmath.ccomplex<f32>
        %c2 = "cmath.mul"(%c, %c) : (!cmath.ccomplex<f32>, !cmath.ccomplex<f32>) -> !cmath.ccomplex<f32>
        // CHECK: %{{.*}} = "cmath.norm"(%{{.*}}) : (!cmath.ccomplex<f32>) -> f32
        %resnorm = "cmath.norm"(%c2) : (!cmath.ccomplex<f32>) -> f32
        // CHECK: %{{.*}} = "cmath.get_real"(%{{.*}}) : (!cmath.ccomplex<f32>) -> f32
        %res_real = "cmath.get_real"(%c2) : (!cmath.ccomplex<f32>) -> f32
        // CHECK: %{{.*}} = "cmath.get_imaginary"(%{{.*}}) : (!cmath.ccomplex<f32>) -> f32
        %res_im = "cmath.get_imaginary"(%c2) : (!cmath.ccomplex<f32>) -> f32
        return
    }
}