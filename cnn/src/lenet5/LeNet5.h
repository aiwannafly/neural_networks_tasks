#ifndef CNN_LENET5_H
#define CNN_LENET5_H

#include "NeuralNetwork.h"
#include "../layers/convolution/ConvolutionLayer.h"
#include "../layers/pooling/MaxPoolingLayer.h"
#include "../layers/PerceptronLayer.h"

namespace cnn {
    /*
     * This NN accepts 28x28 length vectors, returns ${output_size}
     * probabilities of belonging to one of a class.
     */
    class LeNet5: public NeuralNetwork {
    public:
        explicit LeNet5(size_t output_size);

        Eigen::VectorXf predict(const Eigen::VectorXf &input) override;

        ~LeNet5() override;

    private:
        size_t output_size;
        std::vector<CNNLayer *> cnn_layers;
        std::vector<perceptron::PerceptronLayer *> dense_layers;
        float learning_rate = 0.001;
    };
}

#endif //CNN_LENET5_H
