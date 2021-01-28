// RUN: dyn-opt %s | dyn-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar(%{{.*}}: !complex.real, %{{.*}}: !complex.real) {
    func @bar(%re: !complex.real, %im: !complex.real) {
        // CHECK: %{{.*}} = "complex.make_complex"(%{{.*}}, %{{.*}}) : (!complex.real, !complex.real) -> !complex.complex
        %c = "complex.make_complex"(%re, %im) : (!complex.real, !complex.real) -> !complex.complex
        // CHECK: %{{.*}} = "complex.mul"(%{{.*}}, %{{.*}}) : (!complex.complex, !complex.complex) -> !complex.complex
        %c2 = "complex.mul"(%c, %c) : (!complex.complex, !complex.complex) -> !complex.complex
        // CHECK: %{{.*}} = "complex.norm"(%{{.*}}) : (!complex.complex) -> !complex.real
        %resnorm = "complex.norm"(%c2) : (!complex.complex) -> !complex.real
        // CHECK: %{{.*}}:2 = "complex.components"(%{{.*}}) : (!complex.complex) -> (!complex.real, !complex.real)
        %res2:2 = "complex.components"(%c2) : (!complex.complex) -> (!complex.real, !complex.real)
        return
    }
}