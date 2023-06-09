#ifndef PERCEPTRON_NEURALNETWORK_H
#define PERCEPTRON_NEURALNETWORK_H

#include "../eigen/Eigen/Core"

namespace perceptron {
    typedef struct {
        Eigen::VectorXf sample;
        Eigen::VectorXf target;
    } Example;

    typedef enum {
        REGRESSION, BINARY_CLASSIFICATION
    } Task;

    typedef struct {
        size_t true_positive;
        size_t true_negative;
        size_t false_positive;
        size_t false_negative;
    } BinaryClassificationScore;

    class NeuralNetwork {
    public:
        virtual Eigen::VectorXf predict(const Eigen::VectorXf &input) = 0;

        virtual void train(const std::vector<Example> &examples) = 0;

        virtual float get_mse(const std::vector<Example> &examples) = 0;

        virtual float get_mae(const std::vector<Example> &examples) = 0;

        virtual float get_rscore(const std::vector<Example> &examples) = 0;

        virtual BinaryClassificationScore get_classification_scores(const std::vector<Example> &examples) = 0;

        virtual void set_learning_rate(float new_rate) = 0;

        virtual void set_task_type(Task task) = 0;

        virtual void save_weights(FILE *fp) = 0;

        virtual bool read_weights(FILE *fp) = 0;

        virtual ~NeuralNetwork() = default;
    };
}

#endif //PERCEPTRON_NEURALNETWORK_H
