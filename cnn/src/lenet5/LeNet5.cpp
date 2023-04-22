#include <iostream>
#include "LeNet5.h"
#include "../feature_map/VectorFeatureMap.h"

#define FIRST_CONV_CORE_SIZE (5)
#define SECOND_CONV_CORE_SIZE (5)
#define THIRD_CONV_CORE_SIZE (4)
#define POOLING_SIZE (4)
#define LAST_CONV_LAYER_SIZE (26)
#define SIGMOID_PARAM (1.0)
#define IMG_SIZE (28)

namespace cnn {
    LeNet5::LeNet5(size_t output_size) {
        this->output_size = output_size;
//        cnn_layers.push_back(new ConvolutionLayer(FIRST_CONV_CORE_SIZE, 4));
//        cnn_layers.push_back(new MaxPoolingLayer(POOLING_SIZE));
//        cnn_layers.push_back(new ConvolutionLayer(SECOND_CONV_CORE_SIZE, 3));
//        cnn_layers.push_back(new MaxPoolingLayer(POOLING_SIZE));
//        cnn_layers.push_back(new ConvolutionLayer(THIRD_CONV_CORE_SIZE));
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
        FeatureMap *inputMap = new VectorFeatureMap(IMG_SIZE, IMG_SIZE, input);
        auto *current_maps = new std::vector<FeatureMap *>;
        current_maps->push_back(inputMap);
        for (auto *layer: cnn_layers) {
            auto *output_maps = layer->apply(current_maps);
            delete current_maps;
            current_maps = output_maps;
        }
        Eigen::VectorXf pre_last = Eigen::VectorXf(LAST_CONV_LAYER_SIZE);
        for (size_t i = 0; i < LAST_CONV_LAYER_SIZE; i++) {
            pre_last((int) i) = current_maps->at(i)->getValue(0, 0);
        }
        Eigen::VectorXf current = pre_last;
        for (auto *layer: dense_layers) {
            current = layer->calculate(current);
        }
        return current;
    }
}
