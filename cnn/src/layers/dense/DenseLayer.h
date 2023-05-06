#ifndef PERCEPTRON_LAYER_H
#define PERCEPTRON_LAYER_H

#include "../../eigen.h"

namespace perceptron {
    class DenseLayer {
    public:
        DenseLayer(size_t prev_layer_size, size_t current_layer_size);

        ~DenseLayer();

        size_t getInputSize() const;

        size_t getOutputSize() const;

        Eigen::MatrixXf *getWeights();

        Eigen::VectorXf *getBiases();

        // returns sigma(W * input + bias)
        Eigen::VectorXf apply(const Eigen::VectorXf& input);

    private:
        size_t prev_layer_size;
        size_t current_layer_size;
        Eigen::MatrixXf *weights;
        Eigen::VectorXf *biases;
    };
}

#endif //PERCEPTRON_LAYER_H
