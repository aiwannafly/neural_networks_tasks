#ifndef CNN_LENET5_H
#define CNN_LENET5_H

#include "ConvolutionNeuralNetwork.h"
#include "../layers/convolution/ConvolutionLayer.h"
#include "../layers/pooling/MaxPoolingLayer.h"
#include "../layers/dense/DenseLayer.h"
#include "../layers/softmax/SoftmaxLayer.h"

namespace CNN {
    /*
     * This NN accepts 28x28 length vectors, returns ${output_size}
     * probabilities of belonging to one of a class.
     */
    class LeNet5: public ConvolutionNeuralNetwork {
    public:
        explicit LeNet5(size_t output_size);

        Vector predict(const Tensor3D &input) override;

        void train(const std::vector<Example> &examples) override;

        void trainExample(const Example &example);

        void setLearningRate(float value);

        ~LeNet5() override;

    private:

        size_t output_size;
        std::vector<CNNLayer *> cnn_layers;
        std::vector<perceptron::DenseLayer *> dense_layers;
        SoftmaxLayer *softmax_layer;
        float l_rate = 0.001;

        typedef struct {
            std::vector<Tensor3D> cnn_tensors;
            std::vector<Vector> dense_vectors;
            Vector softmax_output;
        } full_output;

        full_output predictWithFullOutput(const Tensor3D &input);
    };
}

#endif //CNN_LENET5_H
