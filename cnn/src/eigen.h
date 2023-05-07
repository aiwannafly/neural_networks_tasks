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
#define SIGMOID_PARAM (1)
#define TANH_A (4.5)
#define TANH_B (0.5)

#define LOG(a) std::cout << a << std::endl

Vector AsVector(const Tensor3D &input);

Tensor3D AsTensor3D(const Vector &input, int maps, int cols, int rows);

Matrix AsMatrix(const Tensor4D &input);

Tensor4D AsTensor4D(const Matrix &input, int f_count, int maps, int cols, int rows);

void SetTensor3DValue(Tensor3D *tensor, int x, int y, const Vector &value);

Vector GetTensor3DValue(const Tensor3D &tensor, int x, int y);

void AddTensorPart(Tensor3D *dest, int x0, int y0, const Tensor3D &value);

Vector ApplyReLU(Vector input);

void PrintTensor3D(const Tensor3D &tensor);

void PrintTensorDims(const Tensor3D &tensor);

void PrintVector(const Vector &vector);

void PrintMatrix(const Matrix &matrix);

#endif //CNN_EIGEN_H
