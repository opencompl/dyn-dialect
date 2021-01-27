// RUN: dyn-opt %s | dyn-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1.0 : f32
        %1 = constant 1.0 : f32
        // CHECK: %{{.*}} = "complex.make_complex"(%{{.*}}, %{{.*}}) : (f32, f32) -> !complex.complex
        %res1 = "complex.make_complex"(%0, %1) : (f32, f32) -> !complex.complex
        return
    }
}