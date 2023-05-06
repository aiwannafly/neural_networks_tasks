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
        assert(input.dimension(MAPS) == 1);
        assert(input.dimension(ROWS) == IMG_SIZE);
        assert(input.dimension(COLS) == IMG_SIZE);
        Tensor3D current_cnn = input;
        if (PRINT_FORWARD) {
            std::cout << "INPUT: =====================================" << std::endl;
            PrintTensor3D(input);
        }
        for (auto *layer: cnn_layers) {
            current_cnn = layer->apply(current_cnn);
            if (PRINT_FORWARD) {
                std::cout << "NEXT: ==================================" << std::endl;
                PrintTensor3D(current_cnn);
            }
        }
        Eigen::VectorXf dense_input = Eigen::VectorXf(current_cnn.dimension(MAPS));
        assert(current_cnn.dimension(ROWS) == 1);
        assert(current_cnn.dimension(COLS) == 1);
        for (int i = 0; i < current_cnn.dimension(MAPS); i++) {
            dense_input(i) = current_cnn(i, 0, 0) / 10;
        }
        if (PRINT_FORWARD) {
            std::cout << "INPUT: =====================================" << std::endl;
            PrintVector(dense_input);
        }
        Eigen::VectorXf current_dense = dense_input;
        for (auto *layer: dense_layers) {
            current_dense = layer->apply(current_dense);
            if (PRINT_FORWARD) {
                std::cout << "NEXT: ==================================" << std::endl;
                PrintVector(current_dense);
            }
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
//            std::cerr << "back " << i << std::endl;
            tensor_deltas = cnn_layers.at(i)->backprop(input, tensor_deltas, learning_rate);
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
            current_cnn = layer->apply(current_cnn);
            full_output.cnn_tensors.push_back(current_cnn);
        }
        assert(current_cnn.dimension(ROWS) == 1);
        assert(current_cnn.dimension(COLS) == 1);
        Eigen::VectorXf dense_input = Eigen::VectorXf(LAST_CONV_LAYER_SIZE);
        for (int i = 0; i < LAST_CONV_LAYER_SIZE; i++) {
            dense_input(i) = current_cnn(i, 0, 0) / 10;
        }
        Eigen::VectorXf current_dense = dense_input;
        full_output.dense_vectors.push_back(dense_input);
        for (auto *layer: dense_layers) {
            current_dense = layer->apply(current_dense);
            full_output.dense_vectors.push_back(current_dense);
        }
        full_output.softmax_output = softmax_layer->apply(current_dense);
        return full_output;
    }

    Eigen::VectorXf LeNet5::denseLayersBackProp(const LeNet5::full_output& out, const Eigen::VectorXf &expected_output) {
        Eigen::VectorXf loss_deriv = cross_entropy_deriv(expected_output, out.softmax_output);
        Eigen::VectorXf softmax_deriv = mul_inverse(out.softmax_output);
        if (PRINT_BACKWARD) {
            std::cout << "loss_deriv" << std::endl;
            PrintVector(loss_deriv);
            std::cout << "softmax_deriv" << std::endl;
            PrintVector(softmax_deriv);
        }
//        std::cout << softmax_deriv << std::endl;

        Eigen::VectorXf delta_prev;
        Eigen::VectorXf delta_curr;
        for (int layer_id = (int) dense_layers.size() - 1; layer_id >= 0; layer_id--) {
            auto layer = dense_layers.at(layer_id);
            size_t layer_output_size = layer->getOutputSize();
            size_t layer_input_size = layer->getInputSize();
            Eigen::VectorXf layer_input = out.dense_vectors.at(layer_id);
            Eigen::VectorXf layer_output = out.dense_vectors.at(layer_id + 1);
            Eigen::VectorXf act_deriv = sigmoid_deriv(layer_output);
            if (PRINT_BACKWARD) {
                std::cout << "layer_output" << std::endl;
                PrintVector(layer_output);
                std::cout << "layer_input" << std::endl;
                PrintVector(layer_input);
                std::cout << "act_deriv" << std::endl;
                PrintVector(act_deriv);
            }
            delta_curr = Eigen::VectorXf(layer_output_size);
            Eigen::MatrixXf *weights = layer->getWeights();
            Eigen::VectorXf *biases = layer->getBiases();
            if (layer_id == (int) dense_layers.size() - 1) {
                for (int j = 0; j < layer_output_size; j++) {
                    delta_curr(j) = -loss_deriv(j) * softmax_deriv(j) * act_deriv(j);
                    float common_err = learning_rate * delta_curr(j);
                    for (int k = 0; k < layer_input_size; k++) {
                        float err = common_err * layer_input(k);
                        (*weights)(j, k) += err;
                    }
                    (*biases)(j) += common_err;
                }
            } else {
                auto next_layer = dense_layers.at(layer_id + 1);
                Eigen::MatrixXf *next_weights = next_layer->getWeights();
                for (int j = 0; j < layer_output_size; j++) {
                    Eigen::VectorXf prev_weights = next_weights->col(j);
                    float scalar_prod = prev_weights.dot(delta_prev);
                    delta_curr(j) = scalar_prod * act_deriv(j);
                    float common_err = learning_rate * delta_curr(j);
                    for (int k = 0; k < layer_input_size; k++) {
                        float err = common_err * layer_input(k);
                        (*weights)(j, k) += err;
                    }
                    (*biases)(j) += common_err;
                }
            }
            delta_prev = delta_curr;
            if (PRINT_BACKWARD) {
                std::cout << "deltas" << std::endl;
                PrintVector(delta_curr);
            }
        }
        auto *first_layer = dense_layers.at(0);
        delta_curr = Eigen::VectorXf(first_layer->getInputSize());
        for (int j = 0; j < first_layer->getInputSize(); j++) {
            Eigen::VectorXf curr_weights = first_layer->getWeights()->col(j);
            float scalar_prod = curr_weights.dot(delta_prev);
            delta_curr(j) = scalar_prod;
        }
        if (PRINT_BACKWARD) {
            std::cout << "deltas" << std::endl;
            PrintVector(delta_curr);
        }
        return delta_curr;
    }
}
