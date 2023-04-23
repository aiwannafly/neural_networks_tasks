#include <iostream>
#include "LeNet5.h"

#define FIRST_CONV_CORE_SIZE (5)
#define SECOND_CONV_CORE_SIZE (5)
#define THIRD_CONV_CORE_SIZE (4)
#define POOLING_SIZE (2)
#define LAST_CONV_LAYER_SIZE (26)
#define SIGMOID_PARAM (1.0)
#define IMG_SIZE (28)

namespace cnn {
    LeNet5::LeNet5(size_t output_size) {
        this->output_size = output_size;

        cnn_layers.push_back(new ConvolutionLayer(FIRST_CONV_CORE_SIZE, 4, 1));
        cnn_layers.push_back(new MaxPoolingLayer(POOLING_SIZE));
        cnn_layers.push_back(new ConvolutionLayer(SECOND_CONV_CORE_SIZE, 12, 4));
        cnn_layers.push_back(new MaxPoolingLayer(POOLING_SIZE));
        cnn_layers.push_back(new ConvolutionLayer(THIRD_CONV_CORE_SIZE, 26, 12));
        dense_layers.push_back(new perceptron::PerceptronLayer(LAST_CONV_LAYER_SIZE, output_size, SIGMOID_PARAM));
    }

    LeNet5::~LeNet5() {
        for (auto *layer: cnn_layers) {
            delete layer;
        }
        for (auto *layer: dense_layers) {
            delete layer;
        }
    }

    Eigen::VectorXf LeNet5::predict(const Eigen::VectorXf &input) {
        assert(input.size() == IMG_SIZE * IMG_SIZE);
        Tensor3D current(1, IMG_SIZE, IMG_SIZE);
        for (int i = 0; i < IMG_SIZE; i++) {
            for (int j = 0; j < IMG_SIZE; j++) {
                current(0, i, j) = input(j + i * IMG_SIZE);
            }
        }
        std::cout << current.dimension(SLICES) << std::endl;
        std::cout << current.dimension(ROWS) << std::endl;
        std::cout << current.dimension(COLS) << std::endl;
        for (auto *layer: cnn_layers) {
            current = layer->apply(current);
//            std::cout << current.dimension(SLICES) << "x" << current.dimension(ROWS) << "x" << current.dimension(COLS) << std::endl;
        }
        Eigen::VectorXf pre_last = Eigen::VectorXf(LAST_CONV_LAYER_SIZE);
        for (int i = 0; i < LAST_CONV_LAYER_SIZE; i++) {
            pre_last(i) = current(i, 0, 0);
        }
        Eigen::VectorXf currentV = pre_last;
        for (auto *layer: dense_layers) {
            currentV = layer->apply(currentV);
        }
        return currentV;
    }
}
