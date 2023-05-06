#include "eigen.h"

#include <iostream>

void PrintTensor3D(const Tensor3D &tensor) {
    for (int l = 0; l < tensor.dimension(MAPS); l++) {
        std::cout << "> slice " << l << std::endl;
        for (int i = 0; i < tensor.dimension(ROWS); i++) {
            for (int j = 0; j < tensor.dimension(COLS); j++) {
                std::cout << tensor(l, i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
}


void PrintTensorDims(const Tensor3D &tensor) {
    std::cout << tensor.dimension(0) << "x" << tensor.dimension(1) << "x" << tensor.dimension(2) << std::endl;
}

void PrintVector(const Eigen::VectorXf &vector) {
    std::cout << "(";
    for (int i = 0; i < vector.size() - 1; i++) {
        std::cout << vector(i) << ", ";
    }
    std::cout << vector(vector.size() - 1) << ")" << std::endl;
}


Eigen::VectorXf toVector(const Tensor3D &input) {
    int maps = (int) input.dimension(MAPS);
    int rows = (int) input.dimension(ROWS);
    int cols = (int) input.dimension(COLS);
    auto res = Eigen::VectorXf(maps * rows * cols);
    for (int map = 0; map < maps; map++) {
        int mapOffset = map * rows * cols;
        for (int row = 0; row < rows; row++) {
            int rowOffset = row * cols;
            for (int col = 0; col < cols; col++) {
                res(mapOffset + rowOffset + col) = input(map, row, col);
            }
        }
    }
    return res;
}

Tensor3D toTensor3D(const Eigen::VectorXf &input, int maps, int cols, int rows) {
    assert(input.size() == maps * rows * cols);
    auto res = Tensor3D(maps, rows, cols);
    for (int map = 0; map < maps; map++) {
        int mapOffset = map * rows * cols;
        for (int row = 0; row < rows; row++) {
            int rowOffset = row * cols;
            for (int col = 0; col < cols; col++) {
                res(map, row, col) = input(mapOffset + rowOffset + col);
            }
        }
    }
    return res;
}

Eigen::MatrixXf toMatrix(const Tensor4D &input) {
    int f_count = (int) input.dimension(0);
    int maps = (int) input.dimension(MAPS + 1);
    int rows = (int) input.dimension(ROWS + 1);
    int cols = (int) input.dimension(COLS + 1);
    auto res = Eigen::MatrixXf(f_count, maps * rows * cols);
    for (int f_curr = 0; f_curr < f_count; f_curr++) {
        for (int map = 0; map < maps; map++) {
            int mapOffset = map * rows * cols;
            for (int row = 0; row < rows; row++) {
                int rowOffset = row * cols;
                for (int col = 0; col < cols; col++) {
                    res(f_curr, mapOffset + rowOffset + col) = input(f_curr, map, row, col);
                }
            }
        }
    }
    return res;
}

Tensor4D toTensor4D(const Eigen::MatrixXf &input, int f_count, int maps, int cols, int rows) {
    assert(input.size() == f_count * maps * rows * cols);
    auto res = Tensor4D(f_count, maps, rows, cols);
    for (int f_curr = 0; f_curr < f_count; f_curr++) {
        for (int map = 0; map < maps; map++) {
            int mapOffset = map * rows * cols;
            for (int row = 0; row < rows; row++) {
                int rowOffset = row * cols;
                for (int col = 0; col < cols; col++) {
                    res(f_curr, map, row, col) = input(f_curr, mapOffset + rowOffset + col);
                }
            }
        }
    }
    return res;
}
