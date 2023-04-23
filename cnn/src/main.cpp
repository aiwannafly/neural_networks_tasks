#include <iostream>

#include "lenet5/LeNet5.h"

#include "eigen.h"

#define IMG_SIZE (28)

/*
 * TODO:
 *  - implement back propagation
 */
int main() {
    cnn::LeNet5 nn = cnn::LeNet5(10);
    Eigen::VectorXf input = Eigen::VectorXf(IMG_SIZE * IMG_SIZE);
    input.setRandom();
    Tensor3D tensorInput(1, IMG_SIZE, IMG_SIZE);
    for (int i = 0; i < IMG_SIZE; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
            tensorInput(0, i, j) = input(j + i * IMG_SIZE);
        }
    }
    auto output = nn.predict(tensorInput);
    std::cout << output << std::endl;
    return 0;
}

//    Eigen::Tensor<float, 3> m(3, 10, 10);          //Initialize
//    m.setConstant(3);                               //Set random values
//    m(0, 0, 0) = 7;
//    m(0, 1, 1) = 9;
//    m(0, 5, 5) = 8;
//    Eigen::Tensor<float, 3> a(1, 4, 4);
//    a = m;

////    std::array<size_t, 3> offset = {0, 0, 0};         //Starting point
////    std::array<size_t, 3> extent = {1, 5, 5};       //Finish point
//    std::array<size_t, 2> shape2 = {5, 5};         //Shape of desired rank-2 tensor (matrix)
////    Eigen::Tensor<float, 3> core(3, 5, 5);
////    core.setConstant(2);
//
//    std::array<size_t, 3> offset = {0, 2, 1};         //Starting point
//    std::array<size_t, 3> extent = {1, 5, 5};       //Finish point
////    Eigen::Tensor<float, 3> ans(3, 5, 5);
////    Eigen::Tensor<float, 0> coreSum = core.sum();
////    ans = m.slice(offset2, extent2) * core;
////    float max = m.slice(offset, extent).maximum().NumDimensions;
////    std::cout << max << std::endl;
//
//    Eigen::Tensor<float, 3> slice = m.slice(offset, extent);
//    Eigen::Tensor<float, 0> b = slice.maximum();
//    float max = b(0);
//    std::cout << max << std::endl;
//    std::cout << slice.reshape(shape2) << std::endl;  //Extract slice