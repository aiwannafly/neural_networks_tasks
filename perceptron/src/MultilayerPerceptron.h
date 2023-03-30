#ifndef PERCEPTRON_MULTILAYERPERCEPTRON_H
#define PERCEPTRON_MULTILAYERPERCEPTRON_H

#include <vector>
#include "../eigen/Eigen/Eigen"
#include "NeuralNetwork.h"
#include "Layer.h"

namespace perceptron {
    class MultilayerPerceptron : public NeuralNetwork {
    public:
        /*
         * layer_sizes vector should contain sizes of all layers in NN,
         * including entry layer, hidden layers and output layer
         */
        explicit MultilayerPerceptron(const std::vector<size_t>& layer_sizes);

        Eigen::VectorXf predict(const Eigen::VectorXf &input) override;

        void train(const std::vector<Example> &examples) override;

        float get_err(const std::vector<Example> &examples) override;

        void set_learning_rate(float new_rate) override;

        void save_weights(FILE *fp) override;

        bool read_weights(FILE *fp) override;

        ~MultilayerPerceptron() override;

    private:
        /*
         * calculates the gradient of a cost function, then uses it for
         * changing weights to reduce difference between expected_output
         * and real output
         */
        void train_example(const Eigen::VectorXf &input, const Eigen::VectorXf &expected_output);

        float get_example_err(const Eigen::VectorXf &input, const Eigen::VectorXf &expected_output);

        std::vector<Eigen::VectorXf> *get_interim_results(const Eigen::VectorXf &input);

        std::vector<Layer*> *layers;
        float learning_rate = 0.001;
    };
}

#endif //PERCEPTRON_MULTILAYERPERCEPTRON_H
