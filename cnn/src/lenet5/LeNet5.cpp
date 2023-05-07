#include <iostream>
#include "LeNet5.h"
#include "../common/functions.h"

#define FIRST_CONV_CORE_SIZE (5)
#define SECOND_CONV_CORE_SIZE (5)
#define THIRD_CONV_CORE_SIZE (4)
#define POOLING_SIZE (2)
#define LAST_CONV_LAYER_SIZE (26)
#define IMG_SIZE (28)

namespace CNN {
    LeNet5::LeNet5(size_t output_size) {
        this->output_size = output_size;
        cnn_layers.push_back(new ConvolutionLayer(FIRST_CONV_CORE_SIZE, 4, 1));
        cnn_layers.push_back(new MaxPoolingLayer(POOLING_SIZE));
        cnn_layers.push_back(new ConvolutionLayer(SECOND_CONV_CORE_SIZE, 12, 4));
        cnn_layers.push_back(new MaxPoolingLayer(POOLING_SIZE));
        cnn_layers.push_back(new ConvolutionLayer(THIRD_CONV_CORE_SIZE, 26, 12));
        dense_layers.push_back(new perceptron::DenseLayer(LAST_CONV_LAYER_SIZE, output_size));
//        dense_layers.push_back(new perceptron::DenseLayer(LAST_CONV_LAYER_SIZE, 16));
//        dense_layers.push_back(new perceptron::DenseLayer(16, 12));
//        dense_layers.push_back(new perceptron::DenseLayer(12, output_size));
        softmax_layer = new SoftmaxLayer();
    }

    LeNet5::~LeNet5() {
        for (auto *layer: cnn_layers) {
            delete layer;
        }
        for (auto *layer: dense_layers) {
            delete layer;
        }
        delete softmax_layer;
    }

    Vector LeNet5::predict(const Tensor3D &input) {
        return predictWithFullOutput(input).softmax_output;
    }

    void LeNet5::train(const std::vector<Example> &examples) {
        std::vector<Matrix> weightsDeltas;

        for (const auto &e: examples) {
            trainExample(e);
        }
    }

    void LeNet5::trainExample(const Example &example) {
        assert(example.expected_output.size() == output_size);
        full_output out = predictWithFullOutput(example.sample);

        // score initial deltas
        Vector loss_deriv = CrossEntropyDeriv(example.expected_output, out.softmax_output);
//        LOG("dense output:");
//        PrintVector(out.dense_vectors.back());
//        LOG("loss_deriv:");
//        PrintVector(loss_deriv);
        Vector softmax_deriv = SoftmaxLayer::deriv(out.softmax_output);
//        LOG("softmax_deriv:");
//        PrintVector(softmax_deriv);
        Vector act_deriv = perceptron::DenseLayer::act_deriv(out.dense_vectors.back());
//        LOG("act_deriv:");
//        PrintVector(act_deriv);
        Vector curr_dense_deltas = Vector(out.dense_vectors.back().size());
        for (int j = 0; j < out.dense_vectors.back().size(); j++) {
            curr_dense_deltas(j) = -loss_deriv(j) * softmax_deriv(j) * act_deriv(j);
        }

        // back propagation in dense layers
        for (int i = (int) dense_layers.size() - 1; i >= 0; i--) {
            auto layer_input = out.dense_vectors.at(i);
            curr_dense_deltas = dense_layers.at(i)->backprop(layer_input, curr_dense_deltas, l_rate);
        }
        //return;
        // back propagation in cnn layers
        Tensor3D curr_cnn_deltas(curr_dense_deltas.size(), 1, 1);
        for (int i = 0; i < curr_dense_deltas.size(); i++) {
            curr_cnn_deltas(i, 0, 0) = -curr_dense_deltas(i);
        }
        for (int i = (int) cnn_layers.size() - 1; i >= 0; i--) {
            auto layer_input = out.cnn_tensors.at(i);
            curr_cnn_deltas = cnn_layers.at(i)->backprop(layer_input, curr_cnn_deltas, l_rate);
        }
    }

    LeNet5::full_output LeNet5::predictWithFullOutput(const Eigen::Tensor<float, 3> &input) {
        assert(input.dimension(MAPS) == 1);
        assert(input.dimension(ROWS) == IMG_SIZE);
        assert(input.dimension(COLS) == IMG_SIZE);
        auto full_output = LeNet5::full_output();
        full_output.cnn_tensors.push_back(input);
        Tensor3D current_cnn = input;
        for (auto *layer: cnn_layers) {
            current_cnn = layer->forward(current_cnn);
            full_output.cnn_tensors.push_back(current_cnn);
        }
        assert(current_cnn.dimension(ROWS) == 1);
        assert(current_cnn.dimension(COLS) == 1);
        Vector dense_input = Vector(LAST_CONV_LAYER_SIZE);
        for (int i = 0; i < LAST_CONV_LAYER_SIZE; i++) {
            dense_input(i) = current_cnn(i, 0, 0);
        }
        Vector current_dense = dense_input;
        full_output.dense_vectors.push_back(dense_input);
        for (auto *layer: dense_layers) {
            current_dense = layer->forward(current_dense);
            full_output.dense_vectors.push_back(current_dense);
        }
        full_output.softmax_output = CNN::SoftmaxLayer::forward(current_dense);
        return full_output;
    }
}
