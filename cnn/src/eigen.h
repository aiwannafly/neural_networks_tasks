#ifndef CNN_EIGEN_H
#define CNN_EIGEN_H

//#include <eigen3/Eigen/Eigen>
//#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "../../perceptron/eigen/Eigen/Eigen"
#include "../../perceptron/eigen/unsupported/Eigen/CXX11/Tensor"

#define Tensor3D Eigen::Tensor<float, 3>
#define Tensor4D Eigen::Tensor<float, 4>
#define LongsTensor3D Eigen::Tensor<int, 3>

#define MAPS (0)
#define ROWS (1)
#define COLS (2)
#define PRINT_FORWARD (0)
#define PRINT_BACKWARD (0)
#define SIGMOID_PARAM (0.1)

void PrintTensor(const Tensor3D &tensor);

void PrintTensorDims(const Tensor3D &tensor);

void PrintVector(const Eigen::VectorXf &vector);

#endif //CNN_EIGEN_H
