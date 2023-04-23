#include "SoftmaxLayer.h"

Eigen::VectorXf cnn::SoftmaxLayer::apply(const Eigen::VectorXf &input) {
    size_t size = input.size();
    Eigen::VectorXf result = Eigen::VectorXf(size);
    float denominator = 0;
    for (int i = 0; i < size; i++) {
        denominator += std::exp(input(i));
    }
    for (int i = 0; i < size; i++) {
        result(i) = std::exp(input(i)) / denominator;
    }
    return result;
}
