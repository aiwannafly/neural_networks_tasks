#ifndef PERCEPTRON_LAYER_H
#define PERCEPTRON_LAYER_H

#include "../eigen.h"

namespace perceptron {
    class PerceptronLayer {
    public:
        PerceptronLayer(size_t prev_layer_size, size_t current_layer_size, float sigmoid_param);

        ~PerceptronLayer();

        size_t getPrevLayerSize() const;

        size_t getCurrentLayerSize() const;

        Eigen::MatrixXf *getWeights();

        Eigen::VectorXf *getBiases();

        // returns sigma(W * input + bias)
        Eigen::VectorXf apply(const Eigen::VectorXf& input);

    private:
       float sigmoid_param;
        size_t prev_layer_size;
        size_t current_layer_size;
        Eigen::MatrixXf *weights;
        Eigen::VectorXf *biases;
    };
}

#endif //PERCEPTRON_LAYER_H
