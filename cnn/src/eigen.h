#ifndef CNN_EIGEN_H
#define CNN_EIGEN_H

#include <eigen3/Eigen/Eigen>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
//#include "../../perceptron/eigen/Eigen/Eigen"
//#include "../../perceptron/eigen/unsupported/Eigen/CXX11/Tensor"

#define Tensor3D Eigen::Tensor<float, 3>
#define Tensor4D Eigen::Tensor<float, 4>

#define SLICES (0)
#define ROWS (1)
#define COLS (2)
#define CORES (3)

#endif //CNN_EIGEN_H
