#include <iostream>

#include "NeuralNetwork.h"

int main() {
    std::vector<size_t> sizes = {4, 4, 3, 3};
    auto nn = new perceptron::NeuralNetwork(sizes);
    Eigen::VectorXf input(4);
    input.setConstant(3);
    Eigen::VectorXf output(3);
    output(0) = 0.2;
    output(1) = 0.6;
    output(2) = 0.4;

    for (int i = 0; i < 10000; i++) {
        nn->train(input, output);
    }
    delete nn;
    return 0;
}
