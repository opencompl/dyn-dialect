// RUN: dyn-opt %s --irdl-file=%S/cmath.irdl | dyn-opt --irdl-file=%S/cmath.irdl | FileCheck %s

module {
    // CHECK-LABEL: func @bar(%{{.*}}: !cmath.real, %{{.*}}: !cmath.real) {
    func @bar(%re: !cmath.real, %im: !cmath.real) {
        // CHECK: %{{.*}} = "cmath.make_complex"(%{{.*}}, %{{.*}}) : (!cmath.real, !cmath.real) -> !cmath.ccomplex<f32>
        %c = "cmath.make_complex"(%re, %im) : (!cmath.real, !cmath.real) -> !cmath.ccomplex<f32>
        // CHECK: %{{.*}} = "cmath.mul"(%{{.*}}, %{{.*}}) : (!cmath.ccomplex<f32>, !cmath.ccomplex<f32>) -> !cmath.ccomplex<f32>
        %c2 = "cmath.mul"(%c, %c) : (!cmath.ccomplex<f32>, !cmath.ccomplex<f32>) -> !cmath.ccomplex<f32>
        // CHECK: %{{.*}} = "cmath.norm"(%{{.*}}) : (!cmath.ccomplex<f32>) -> !cmath.real
        %resnorm = "cmath.norm"(%c2) : (!cmath.ccomplex<f32>) -> !cmath.real
        // CHECK: %{{.*}} = "cmath.get_real"(%{{.*}}) : (!cmath.ccomplex<f32>) -> !cmath.real
        %res_real = "cmath.get_real"(%c2) : (!cmath.ccomplex<f32>) -> !cmath.real
        // CHECK: %{{.*}} = "cmath.get_imaginary"(%{{.*}}) : (!cmath.ccomplex<f32>) -> !cmath.real
        %res_im = "cmath.get_imaginary"(%c2) : (!cmath.ccomplex<f32>) -> !cmath.real
        return
    }
}