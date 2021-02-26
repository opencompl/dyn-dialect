// RUN: dyn-opt %s --irdl-file=%S/cmath.irdl | dyn-opt --irdl-file=%S/cmath.irdl | FileCheck %s

module {
    // CHECK-LABEL: func @bar(%{{.*}}: !cmath.real, %{{.*}}: !cmath.real) {
    func @bar(%re: !cmath.real, %im: !cmath.real) {
        // CHECK: %{{.*}} = "cmath.make_complex"(%{{.*}}, %{{.*}}) : (!cmath.real, !cmath.real) -> tuple<!cmath.real, !cmath.real>
        %c = "cmath.make_complex"(%re, %im) : (!cmath.real, !cmath.real) -> !cmath.ccomplex
        // CHECK: %{{.*}} = "cmath.mul"(%{{.*}}, %{{.*}}) : (tuple<!cmath.real, !cmath.real>, tuple<!cmath.real, !cmath.real>) -> tuple<!cmath.real, !cmath.real>
        %c2 = "cmath.mul"(%c, %c) : (!cmath.ccomplex, !cmath.ccomplex) -> !cmath.ccomplex
        // CHECK: %{{.*}} = "cmath.norm"(%{{.*}}) : (tuple<!cmath.real, !cmath.real>) -> !cmath.real
        %resnorm = "cmath.norm"(%c2) : (!cmath.ccomplex) -> !cmath.real
        // CHECK: %{{.*}} = "cmath.get_real"(%{{.*}}) : (tuple<!cmath.real, !cmath.real>) -> !cmath.real
        %res_real = "cmath.get_real"(%c2) : (!cmath.ccomplex) -> !cmath.real
        // CHECK: %{{.*}} = "cmath.get_imaginary"(%{{.*}}) : (tuple<!cmath.real, !cmath.real>) -> !cmath.real
        %res_im = "cmath.get_imaginary"(%c2) : (!cmath.ccomplex) -> !cmath.real
        return
    }
}