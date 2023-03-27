#include "Layer.h"

#include <cmath>
#include <iostream>

namespace perceptron {

    Layer::Layer(size_t prev_layer_size, size_t current_layer_size) {
        this->current_layer_size = current_layer_size;
        this->prev_layer_size = prev_layer_size;
        biases = new Eigen::VectorXf(current_layer_size);
        weights = new  Eigen::MatrixXf(current_layer_size, prev_layer_size);
        biases->setRandom();
        weights->setRandom();
    }

    Layer::~Layer() {
        delete biases;
        delete weights;
    }

    float sigmoid(float x) {
        return 1 / (1 + std::exp(-x));
    }

    Eigen::VectorXf Layer::calculate(const Eigen::VectorXf& input) {
        Eigen::VectorXf sum = *weights * input + *biases;
        for (int i = 0; i < sum.size(); i++) {
            sum(i) = sigmoid(sum(i));
        }
        // with use of vector op
        return sum;
    }

    Eigen::MatrixXf *Layer::get_weights() {
        return weights;
    }

    Eigen::VectorXf *Layer::get_biases() {
        return biases;
    }

    size_t Layer::get_prev_layer_size() const {
        return prev_layer_size;
    }

    size_t Layer::get_current_layer_size() const {
        return current_layer_size;
    }
}
