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
        dense_layers.push_back(new perceptron::PerceptronLayer(LAST_CONV_LAYER_SIZE, output_size));
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

    Eigen::VectorXf LeNet5::predict(const Tensor3D &input) {
        assert(input.dimension(SLICES) == 1);
        assert(input.dimension(ROWS) == IMG_SIZE);
        assert(input.dimension(COLS) == IMG_SIZE);
        Tensor3D current_cnn = input;
        for (auto *layer: cnn_layers) {
            current_cnn = layer->apply(current_cnn);
        }
        Eigen::VectorXf dense_input = Eigen::VectorXf(current_cnn.dimension(SLICES));
        for (int i = 0; i < current_cnn.dimension(SLICES); i++) {
            dense_input(i) = current_cnn(i, 0, 0) / 10000;
        }
        Eigen::VectorXf current_dense = dense_input;
        for (auto *layer: dense_layers) {
            current_dense = layer->apply(current_dense);
        }
        return softmax_layer->apply(current_dense);
    }

    void LeNet5::train(const std::vector<Example> &examples) {
        for (const auto &e: examples) {
            trainExample(e);
        }
    }

    void LeNet5::trainExample(const Example &example) {
        assert(example.expected_output.size() == output_size);
        full_output out = predictWithFullOutput(example.sample);
        Eigen::VectorXf dense_input_deltas = denseLayersBackProp(out, example.expected_output);
//        std::cout << "DENSE DELTAS" << std::endl;
//        std::cout << dense_input_deltas << std::endl;
        Tensor3D tensor_deltas(dense_input_deltas.size(), 1, 1);
        for (int i = 0; i < dense_input_deltas.size(); i++) {
            tensor_deltas(i, 0, 0) = dense_input_deltas(i);
        }
        for (int i = (int) cnn_layers.size() - 1; i >= 0; i--) {
            auto input = out.cnn_tensors.at(i);
            tensor_deltas = cnn_layers.at(i)->backprop(input, tensor_deltas, learning_rate);
        }
    }

    LeNet5::full_output LeNet5::predictWithFullOutput(const Eigen::Tensor<float, 3> &input) {
        assert(input.dimension(SLICES) == 1);
        assert(input.dimension(ROWS) == IMG_SIZE);
        assert(input.dimension(COLS) == IMG_SIZE);
        auto full_output = LeNet5::full_output();
        full_output.cnn_tensors.push_back(input);
        Tensor3D current = input;
        for (auto *layer: cnn_layers) {
            current = layer->apply(current);
            full_output.cnn_tensors.push_back(current);
        }
        Eigen::VectorXf dense_input = Eigen::VectorXf(LAST_CONV_LAYER_SIZE);
        for (int i = 0; i < LAST_CONV_LAYER_SIZE; i++) {
            dense_input(i) = current(i, 0, 0) / 10000;
        }
        Eigen::VectorXf currentV = dense_input;
        full_output.dense_vectors.push_back(dense_input);
        for (auto *layer: dense_layers) {
            currentV = layer->apply(currentV);
            full_output.dense_vectors.push_back(currentV);
        }
        full_output.softmax_output = softmax_layer->apply(currentV);
        return full_output;
    }

    Eigen::VectorXf LeNet5::denseLayersBackProp(const LeNet5::full_output& out, const Eigen::VectorXf &expected_output) {
        Eigen::VectorXf loss_deriv = cross_entropy_deriv(expected_output, out.softmax_output);
        Eigen::VectorXf softmax_deriv = mul_inverse(out.softmax_output);
//        std::cout << "loss_deriv" << std::endl;
//        std::cout << loss_deriv << std::endl;
//
//        std::cout << "softmax_deriv" << std::endl;
//        std::cout << softmax_deriv << std::endl;

        Eigen::VectorXf delta_prev;
        Eigen::VectorXf delta_curr;
        for (int layer_id = (int) dense_layers.size() - 1; layer_id >= 0; layer_id--) {
            auto layer = dense_layers.at(layer_id);
            size_t layer_output_size = layer->getOutputSize();
            size_t layer_input_size = layer->getInputSize();
            Eigen::VectorXf layer_input = out.dense_vectors.at(layer_id);
            Eigen::VectorXf layer_output = out.dense_vectors.at(layer_id + 1);

//            std::cout << "layer_output" << std::endl;
//            std::cout << layer_output << std::endl;
            Eigen::VectorXf tan_deriv = tanh_deriv(layer_output);
//            std::cout << "layer_input" << std::endl;
//            std::cout << layer_input << std::endl;
//            std::cout << "tan_deriv" << std::endl;
//            std::cout << tan_deriv << std::endl;
            delta_curr = Eigen::VectorXf(layer_output_size);
            Eigen::MatrixXf *weights = dense_layers.at(layer_id)->getWeights();
            Eigen::VectorXf *biases = dense_layers.at(layer_id)->getBiases();
            if (layer_id == (int) dense_layers.size() - 1) {
                for (int j = 0; j < layer_output_size; j++) {
                    delta_curr(j) = -loss_deriv(j) * softmax_deriv(j) * tan_deriv(j);
                    float common_err = learning_rate * delta_curr(j);
                    for (int k = 0; k < layer_input_size; k++) {
                        float err = common_err * layer_input(k);
                        (*weights)(j, k) += err;
                    }
                    float err = common_err;
                    (*biases)(j) += err;
                }
            } else {
                auto next_layer = dense_layers.at(layer_id + 1);
                Eigen::MatrixXf *next_weights = next_layer->getWeights();
                for (int j = 0; j < layer_output_size; j++) {
                    Eigen::VectorXf curr_weights = next_weights->col(j);
                    float scalar_prod = curr_weights.dot(delta_prev);
                    delta_curr(j) = scalar_prod * tan_deriv(j);
                    float common_err = learning_rate * delta_curr(j);
                    for (int k = 0; k < layer_input_size; k++) {
                        float err = common_err * layer_input(k);
                        (*weights)(j, k) += err;
                    }
                    float err = common_err;
                    (*biases)(j) += err;
                }
            }
//            std::cout << "delta_curr" << std::endl;
//            std::cout << delta_curr << std::endl;
            delta_prev = delta_curr;
        }
        auto *first_layer = dense_layers.at(0);
        delta_curr = Eigen::VectorXf(first_layer->getInputSize());
        for (int j = 0; j < first_layer->getInputSize(); j++) {
            Eigen::VectorXf curr_weights = first_layer->getWeights()->col(j);
            float scalar_prod = curr_weights.dot(delta_prev);
            delta_curr(j) = scalar_prod;
        }
        return delta_curr;
    }
}
