#ifndef CNN_EIGEN_H
#define CNN_EIGEN_H

//#include <eigen3/Eigen/Eigen>
//#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "../../perceptron/eigen/Eigen/Eigen"
#include "../../perceptron/eigen/unsupported/Eigen/CXX11/Tensor"

#define Tensor3D Eigen::Tensor<float, 3>
#define Tensor4D Eigen::Tensor<float, 4>
#define LongsTensor3D Eigen::Tensor<int, 3>
#define Matrix Eigen::MatrixXf
#define Vector Eigen::VectorXf

#define MAPS (0)
#define ROWS (1)
#define COLS (2)
#define PRINT_FORWARD (0)
#define PRINT_BACKWARD (0)
#define SIGMOID_PARAM (0.1)

Eigen::VectorXf toVector(const Tensor3D &input);

Tensor3D toTensor3D(const Eigen::VectorXf &input, int maps, int cols, int rows);

Eigen::MatrixXf toMatrix(const Tensor4D &input);

Tensor4D toTensor4D(const Eigen::MatrixXf &input, int f_count, int maps, int cols, int rows);

void setTensor3DValue(Tensor3D *tensor, int x, int y, const Vector &value);

Vector getTensor3DValue(const Tensor3D &tensor, int x, int y);

void addTensorPart(Tensor3D *dest, int x0, int y0, const Tensor3D &value);

Vector applyReLU(Vector input);

void PrintTensor3D(const Tensor3D &tensor);

void PrintTensorDims(const Tensor3D &tensor);

void PrintVector(const Eigen::VectorXf &vector);

#endif //CNN_EIGEN_H
