#ifndef CNN_FUNCTIONS_H
#define CNN_FUNCTIONS_H

#include "../eigen.h"

float Sigmoid(float x);

namespace NN {
    // from [tanh(a), tanh(b), ...] makes [tanh'(a), tanh'(b), ...]
    Vector TanhDeriv(Vector tanh_vector);

    Vector SigmoidDeriv(Vector sigmoid_vector);

    Vector ReLUDeriv(Vector relu_vector);

    Vector SigmoidV(Vector a);

    Vector ReLUV(Vector a);

    Vector TanhV(Vector a);

    float Sigmoid(float x);

    float ReLU(float x);

    float Tanh(float x);

    Vector CrossEntropyDeriv(const Vector &expected_output,
                             const Vector &predicted_output);

    float CrossEntropy(const Vector &expected_output,
                       const Vector &predicted_output);

    Vector MulInverse(Vector vector);
}

#endif //CNN_FUNCTIONS_H
