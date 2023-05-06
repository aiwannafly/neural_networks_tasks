#include "DenseLayer.h"

#include <cmath>
#include <iostream>

#include "../../common/functions.h"

namespace perceptron {

    DenseLayer::DenseLayer(size_t prev_layer_size, size_t current_layer_size) {
        this->current_layer_size = current_layer_size;
        this->prev_layer_size = prev_layer_size;
        biases = new Eigen::VectorXf(current_layer_size);
        weights = new  Eigen::MatrixXf(current_layer_size, prev_layer_size);
//        biases->setZero();
//        weights->setZero();
        biases->setZero();
        weights->setRandom();
    }

    DenseLayer::~DenseLayer() {
        delete biases;
        delete weights;
    }

    Eigen::VectorXf DenseLayer::apply(const Eigen::VectorXf& input) {
        Eigen::VectorXf sum = *weights * input + *biases;
        for (int i = 0; i < sum.size(); i++) {
            sum(i) = sigmoid(sum(i));
        }
        // with use of vector op
        return sum;
    }

    Eigen::MatrixXf *DenseLayer::getWeights() {
        return weights;
    }

    Eigen::VectorXf *DenseLayer::getBiases() {
        return biases;
    }

    size_t DenseLayer::getInputSize() const {
        return prev_layer_size;
    }

    size_t DenseLayer::getOutputSize() const {
        return current_layer_size;
    }
}
