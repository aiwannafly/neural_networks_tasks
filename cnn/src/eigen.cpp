#include "eigen.h"

#include <iostream>

void PrintTensor(const Tensor3D &tensor) {
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
