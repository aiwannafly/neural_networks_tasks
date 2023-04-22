#ifndef PERCEPTRON_NEURALNETWORK_H
#define PERCEPTRON_NEURALNETWORK_H

#include "../eigen.h"

namespace cnn {
    typedef struct {
        Eigen::VectorXf sample;
        Eigen::VectorXf target;
    } Example;

    class NeuralNetwork {
    public:
        virtual Eigen::VectorXf predict(const Eigen::VectorXf &input) = 0;

//        virtual void train(const std::vector<Example> &examples) = 0;
//
//        virtual void setLearningRate(float new_rate) = 0;
//
//        virtual void saveWeights(FILE *fp) = 0;
//
//        virtual bool readWeights(FILE *fp) = 0;

        virtual ~NeuralNetwork() = default;
    };
}

#endif //PERCEPTRON_NEURALNETWORK_H
