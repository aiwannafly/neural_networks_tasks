#ifndef PERCEPTRON_NEURALNETWORK_H
#define PERCEPTRON_NEURALNETWORK_H

#include "eigen/Eigen/Core"

namespace perceptron {
    class NeuralNetwork {
    public:
        virtual Eigen::VectorXf predict(const Eigen::VectorXf &input) = 0;

        virtual void train(std::vector<Eigen::VectorXf> inputs, std::vector<Eigen::VectorXf> expected_outputs) = 0;

        virtual float get_err(std::vector<Eigen::VectorXf> inputs, std::vector<Eigen::VectorXf> expected_outputs) = 0;

        virtual void set_learning_rate(float new_rate) = 0;

        virtual ~NeuralNetwork() = default;
    };
}

#endif //PERCEPTRON_NEURALNETWORK_H
