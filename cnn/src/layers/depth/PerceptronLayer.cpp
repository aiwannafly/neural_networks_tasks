#include "PerceptronLayer.h"

#include <cmath>
#include <iostream>

#include "../../common/functions.h"

namespace perceptron {

    PerceptronLayer::PerceptronLayer(size_t prev_layer_size, size_t current_layer_size) {
        this->current_layer_size = current_layer_size;
        this->prev_layer_size = prev_layer_size;
        biases = new Eigen::VectorXf(current_layer_size);
        weights = new  Eigen::MatrixXf(current_layer_size, prev_layer_size);
        biases->setRandom();
        weights->setRandom();
    }

    PerceptronLayer::~PerceptronLayer() {
        delete biases;
        delete weights;
    }

    Eigen::VectorXf PerceptronLayer::apply(const Eigen::VectorXf& input) {
        Eigen::VectorXf sum = *weights * input + *biases;
        for (int i = 0; i < sum.size(); i++) {
            sum(i) = std::tanh(sum(i));
        }
        // with use of vector op
        return sum;
    }

    Eigen::MatrixXf *PerceptronLayer::getWeights() {
        return weights;
    }

    Eigen::VectorXf *PerceptronLayer::getBiases() {
        return biases;
    }

    size_t PerceptronLayer::getInputSize() const {
        return prev_layer_size;
    }

    size_t PerceptronLayer::getOutputSize() const {
        return current_layer_size;
    }
}
