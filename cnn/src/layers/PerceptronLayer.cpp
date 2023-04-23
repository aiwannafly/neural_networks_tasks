#include "PerceptronLayer.h"

#include <cmath>
#include <iostream>

namespace perceptron {

    PerceptronLayer::PerceptronLayer(size_t prev_layer_size, size_t current_layer_size, float sigmoid_param) {
        this->current_layer_size = current_layer_size;
        this->prev_layer_size = prev_layer_size;
        this->sigmoid_param = sigmoid_param;
        biases = new Eigen::VectorXf(current_layer_size);
        weights = new  Eigen::MatrixXf(current_layer_size, prev_layer_size);
        biases->setRandom();
        weights->setRandom();
    }

    PerceptronLayer::~PerceptronLayer() {
        delete biases;
        delete weights;
    }

    float sigmoid(float x, float sigmoid_param) {
        return 1 / (1 + std::exp(-sigmoid_param * x));
    }

    Eigen::VectorXf PerceptronLayer::apply(const Eigen::VectorXf& input) {
        Eigen::VectorXf sum = *weights * input + *biases;
        for (int i = 0; i < sum.size(); i++) {
            sum(i) = sigmoid(sum(i), sigmoid_param);
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

    size_t PerceptronLayer::getPrevLayerSize() const {
        return prev_layer_size;
    }

    size_t PerceptronLayer::getCurrentLayerSize() const {
        return current_layer_size;
    }
}
