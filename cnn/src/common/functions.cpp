#include "functions.h"

#include <cmath>
#include <utility>

float sigmoid(float x) {
    return 1 / (1 + (float) std::exp(-x * SIGMOID_PARAM));
}

float ReLU(float x) {
    return x > 0 ? x : 0;
}

Eigen::VectorXf tanh_deriv(Eigen::VectorXf tanh_vector) {
    for (int i = 0; i < tanh_vector.size(); i++) {
        tanh_vector(i) = (1 - tanh_vector(i)) * (1 + tanh_vector(i));
    }
    return tanh_vector;
}

Eigen::VectorXf sigmoid_deriv(Eigen::VectorXf sigmoid_vector) {
    return SIGMOID_PARAM * mul_inverse(std::move(sigmoid_vector));
}

Eigen::VectorXf ReLU_deriv(Eigen::VectorXf relu_vector) {
    for (int i = 0; i < relu_vector.size(); i++) {
        if (relu_vector(i) == 0) {
            relu_vector(i) = 0;
        } else {
            relu_vector(i) = 1;
        }
    }
    return relu_vector;
}

Eigen::VectorXf cross_entropy_deriv(const Eigen::VectorXf &expected_output,
                                    const Eigen::VectorXf &predicted_output) {
//    return -2 * (predicted_output - expected_output);
    // classification, cross-entropy deriv
    assert(expected_output.size() == predicted_output.size());
    size_t size = expected_output.size();
    Eigen::VectorXf res = Eigen::VectorXf(size);
    for (int i = 0; i < size; i++) {
        float expected = expected_output(i);
        float probability = predicted_output(i);
        assert(expected == 0 || expected == 1);
        assert(0 <= probability <= 1);
        float loss_deriv = 1;
        const float epsilon = 10e-3;
        if (expected == 0) {
            if (probability == 1) {
                probability -= epsilon;
            }
            loss_deriv /= (1 - probability);
        } else {
            if (probability == 0) {
                probability += epsilon;
            }
            loss_deriv /= -probability;
        }
        res(i) = loss_deriv;
    }
    return res;
}

Eigen::VectorXf mul_inverse(Eigen::VectorXf vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector(i) = (1 - vector(i)) * vector(i);
    }
    return vector;
}
