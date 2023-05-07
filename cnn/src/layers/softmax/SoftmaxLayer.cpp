#include "SoftmaxLayer.h"
#include "../../common/functions.h"

Vector CNN::SoftmaxLayer::forward(const Vector &input) {
    size_t size = input.size();
    Vector result = Vector(size);
    float denominator = 0;
    for (int i = 0; i < size; i++) {
        denominator += std::exp(input(i));
    }
    for (int i = 0; i < size; i++) {
        result(i) = std::exp(input(i)) / denominator;
    }
    return result;
}

Vector CNN::SoftmaxLayer::deriv(const Vector& output) {
    return MulInverse(output);
}
