// RUN: dyn-opt %s -split-input-file -verify-diagnostics

module {
    func @bar(%re: !complex.real) {
        // expected-error@below {{should have exactly 2 operands}}
        %c = "complex.make_complex"(%re) : (!complex.real) -> !complex.complex
        return
    }
}