#ifndef PERCEPTRON_NEURALNETWORK_H
#define PERCEPTRON_NEURALNETWORK_H

#include "../eigen.h"

namespace CNN {
    typedef struct {
        Tensor3D sample;
        Vector expected_output;
    } Example;

    class ConvolutionNeuralNetwork {
    public:
        virtual Vector predict(const Tensor3D &input) = 0;

        virtual void train(const std::vector<Example> &examples) = 0;
//
//        virtual void setLearningRate(float new_rate) = 0;
//
//        virtual void saveWeights(FILE *fp) = 0;
//
//        virtual bool readWeights(FILE *fp) = 0;

        virtual ~ConvolutionNeuralNetwork() = default;
    };
}

#endif //PERCEPTRON_NEURALNETWORK_H
