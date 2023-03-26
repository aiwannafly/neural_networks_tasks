#ifndef PERCEPTRON_NEURALNETWORK_H
#define PERCEPTRON_NEURALNETWORK_H

#include <vector>
#include "eigen/Eigen/Eigen"
#include "Layer.h"

namespace perceptron {
    class NeuralNetwork {
    public:
        /*
         * layer_sizes vector should contain sizes of all layers in NN,
         * including entry layer, hidden layers and output layer
         */
        explicit NeuralNetwork(const std::vector<size_t>& layer_sizes);

        Eigen::VectorXf calculate(const Eigen::VectorXf &input);

        /*
         * calculates the gradient of a cost function, then uses it for
         * changing weights to reduce difference between expected_output
         * and real output
         */
        void train(const Eigen::VectorXf &input, const Eigen::VectorXf &expected_output);

        void set_learning_rate(float new_rate);

        ~NeuralNetwork();

    private:
        std::vector<Eigen::VectorXf> *calculate_and_get_interim_results(const Eigen::VectorXf &input);

        std::vector<Layer*> *layers;
        float learning_rate = 0.001;
    };
}

#endif //PERCEPTRON_NEURALNETWORK_H
