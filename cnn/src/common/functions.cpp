#include "functions.h"

#include <iostream>

#include <cmath>
#include <utility>

float Sigmoid(float x) {
    return 1 / (1 + (float) std::exp(-x * SIGMOID_PARAM));
}

float ReLU(float x) {
    return x > 0 ? x : 0;
}

Vector TanhDeriv(Vector tanh_vector) {
    for (int i = 0; i < tanh_vector.size(); i++) {
        tanh_vector(i) = (TANH_B / TANH_A) * (TANH_A - tanh_vector(i)) * (TANH_A + tanh_vector(i));
    }
    return tanh_vector;
}

float Tanh(float x) {
    return TANH_A * std::tanh(TANH_B * x);
}

Vector SigmoidDeriv(Vector sigmoid_vector) {
    return SIGMOID_PARAM * MulInverse(std::move(sigmoid_vector));
}

Vector ReLUDeriv(Vector relu_vector) {
    for (int i = 0; i < relu_vector.size(); i++) {
        if (relu_vector(i) == 0) {
            relu_vector(i) = 0;
        } else {
            relu_vector(i) = 1;
        }
    }
    return relu_vector;
}

float CrossEntropy(const Vector &expected_output,
                    const Vector &predicted_output) {
    assert(expected_output.size() == predicted_output.size());
    float res = 0;
    for (int i = 0; i < expected_output.size(); i++) {
        res -= expected_output(i) * std::log2(predicted_output(i));
    }
    return res;
}

Vector CrossEntropyDeriv(const Vector &expected_output,
                         const Vector &predicted_output) {
//    return 2 * (predicted_output - expected_output);
    // classification, cross-entropy deriv
    assert(expected_output.size() == predicted_output.size());
    size_t size = expected_output.size();
    Vector res = Vector(size);
    for (int i = 0; i < size; i++) {
        float expected = expected_output(i);
        float p = predicted_output(i);
        assert(expected == 0 || expected == 1);
        assert(0 <= p <= 1);
        float loss_deriv = 1 / (float) std::log(2);
        const float epsilon = 10e-3;
        if (expected == 0) {
            if (p == 1) {
                p -= epsilon;
            }
            loss_deriv /= (1 - p);
        } else {
            if (p == 0) {
                p += epsilon;
            }
            loss_deriv /= -p;
        }
        res(i) = loss_deriv;
    }
    return res;
}

Vector MulInverse(Vector vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector(i) = (1 - vector(i)) * vector(i);
    }
    return vector;
}
