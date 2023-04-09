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
        explicit MultilayerPerceptron(const std::vector<size_t>& layer_sizes, float sigmoid_param);

        Eigen::VectorXf predict(const Eigen::VectorXf &input) override;

        void train(const std::vector<Example> &examples) override;

        float get_mse(const std::vector<Example> &examples) override;

        float get_mae(const std::vector<Example> &examples) override;

        float get_rscore(const std::vector<Example> &examples) override;

        BinaryClassificationScore get_classification_scores(const std::vector<Example> &examples) override;

        void set_learning_rate(float new_rate) override;

        void set_task_type(Task new_task) override;

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

        Eigen::VectorXf get_loss_function_deriv(const Eigen::VectorXf &expected_output, const Eigen::VectorXf &predicted_output);

        float get_example_mse(const Eigen::VectorXf &input, const Eigen::VectorXf &expected_output);

        float get_example_mae(const Eigen::VectorXf &input, const Eigen::VectorXf &expected_output);

        std::vector<Eigen::VectorXf> *get_interim_results(const Eigen::VectorXf &input);

        std::vector<Layer*> *layers;
        std::vector<Layer*> *layers_deltas;
        float learning_rate = 0.001;
        float sigmoid_param;
        Task task = REGRESSION;
    };
}

#endif //PERCEPTRON_MULTILAYERPERCEPTRON_H
