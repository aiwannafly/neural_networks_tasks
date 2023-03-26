#ifndef PERCEPTRON_LAYER_H
#define PERCEPTRON_LAYER_H

#include "eigen/Eigen/Eigen"

namespace perceptron {
    class Layer {
    public:
        Layer(size_t prev_layer_size, size_t current_layer_size);
        ~Layer();

        size_t get_prev_layer_size() const;

        size_t get_current_layer_size() const;

        Eigen::MatrixXf *get_weights();

        Eigen::VectorXf *get_biases();

        // returns sigma(W * input + bias)
        Eigen::VectorXf calculate(const Eigen::VectorXf& input);

        // returns sigma'(W * input + bias)
        Eigen::VectorXf calculate_deriv(const Eigen::VectorXf& input);
    private:
        size_t prev_layer_size;
        size_t current_layer_size;
        Eigen::MatrixXf *weights;
        Eigen::VectorXf *biases;
    };
}

#endif //PERCEPTRON_LAYER_H
