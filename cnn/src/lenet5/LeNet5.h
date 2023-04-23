#ifndef CNN_LENET5_H
#define CNN_LENET5_H

#include "ConvolutionNeuralNetwork.h"
#include "../layers/convolution/ConvolutionLayer.h"
#include "../layers/pooling/MaxPoolingLayer.h"
#include "../layers/depth/PerceptronLayer.h"
#include "../layers/softmax/SoftmaxLayer.h"

namespace cnn {
    /*
     * This NN accepts 28x28 length vectors, returns ${output_size}
     * probabilities of belonging to one of a class.
     */
    class LeNet5: public ConvolutionNeuralNetwork {
    public:
        explicit LeNet5(size_t output_size);

        Eigen::VectorXf predict(const Tensor3D &input) override;

        void train(const std::vector<Example> &examples) override;

        ~LeNet5() override;

    private:
        void trainExample(const Example &example);

        size_t output_size;
        std::vector<CNNLayer *> cnn_layers;
        std::vector<perceptron::PerceptronLayer *> dense_layers;
        SoftmaxLayer *softmax_layer;
        float learning_rate = 0.001;

        typedef struct {
            std::vector<Tensor3D> cnn_tensors;
            std::vector<Eigen::VectorXf> dense_vectors;
            Eigen::VectorXf softmax_output;
        } full_output;

        full_output predictWithFullOutput(const Tensor3D &input);

        Eigen::VectorXf denseLayersBackProp(const LeNet5::full_output& out, const Eigen::VectorXf &expected_output);
    };
}

#endif //CNN_LENET5_H
