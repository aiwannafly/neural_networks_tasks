#include <iostream>

#include "lenet5/LeNet5.h"

//#include <eigen3/Eigen/Eigen>
//#include "../../perceptron/eigen/unsupported/Eigen/CXX11/Tensor"
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
//#include <eigen3/Eigen/Eigen>

int main() {
//    Eigen::Tensor<float, 3> m(3, 10, 10);          //Initialize
//    m.setConstant(3);                               //Set random values
//    m(0, 0, 0) = 7;
//    m(0, 1, 1) = 9;
//    std::array<long, 3> offset = {0, 0, 0};         //Starting point
//    std::array<long, 3> extent = {1, 5, 5};       //Finish point
//    std::array<long, 2> shape2 = {5, 5};         //Shape of desired rank-2 tensor (matrix)
//    Eigen::Tensor<float, 3> core(3, 5, 5);
//    core.setConstant(2);
//
//    std::array<long, 3> offset2 = {0, 0, 0};         //Starting point
//    std::array<long, 3> extent2 = {3, 5, 5};       //Finish point
//    Eigen::Tensor<float, 3> ans(3, 5, 5);
//    Eigen::Tensor<float, 0> coreSum = core.sum();
//    ans = m.slice(offset2, extent2) * core;
//    std::cout << ans.slice(offset, extent).reshape(shape2) << std::endl;  //Extract slice

//    cnn::LeNet5 nn = cnn::LeNet5(10);
//    Eigen::VectorXf input = Eigen::VectorXf(28 * 28);
//    input.setRandom();
//    auto output = nn.predict(input);
//    std::cout << output << std::endl;
    return 0;
}
