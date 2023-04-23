#ifndef CNN_FUNCTIONS_H
#define CNN_FUNCTIONS_H

#include "../eigen.h"

float sigmoid(float x, float param);

// from [tanh(a), tanh(b), ...] makes [tanh'(a), tanh'(b), ...]
Eigen::VectorXf tanh_deriv(Eigen::VectorXf tanh_vector);

Eigen::VectorXf sigmoid_deriv(Eigen::VectorXf sigmoid_vector);

float ReLU(float x);

Eigen::VectorXf cross_entropy_deriv(const Eigen::VectorXf &expected_output,
                                    const Eigen::VectorXf &predicted_output);

Eigen::VectorXf mul_inverse(Eigen::VectorXf vector);

#endif //CNN_FUNCTIONS_H
