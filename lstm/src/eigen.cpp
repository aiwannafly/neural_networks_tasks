#include "eigen.h"

#include <iostream>

#include "common/functions.h"

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

void PrintVector(const Vector &vector) {
    std::cout << "(";
    for (int i = 0; i < vector.size() - 1; i++) {
        std::cout << vector(i) << ", ";
    }
    std::cout << vector(vector.size() - 1) << ")" << std::endl;
}

void PrintMatrix(const Matrix &matrix) {
    std::cout << matrix << std::endl;
}


Vector AsVector(const Tensor3D &input) {
    int maps = (int) input.dimension(MAPS);
    int rows = (int) input.dimension(ROWS);
    int cols = (int) input.dimension(COLS);
    auto res = Vector(maps * rows * cols);
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

void SetTensor3DValue(Tensor3D *tensor, int x, int y, const Vector &value) {
    int maps = (int) tensor->dimension(0);
    assert(maps == value.size());
    for (int i = 0; i < maps; i++) {
        (*tensor)(i, y, x) = value(i);
    }
}

Vector GetTensor3DValue(const Tensor3D &tensor, int x, int y) {
    int maps = (int) tensor.dimension(0);
    Vector values = Vector(maps);
    for (int i = 0; i < maps; i++) {
        values(i) = tensor(i, y, x);
    }
    return values;
}

void AddTensorPart(Tensor3D *dest, int x0, int y0, const Tensor3D &value) {
    assert(dest->dimension(MAPS) == value.dimension(MAPS));
    for (int y = y0; y < y0 + value.dimension(ROWS); y++) {
        for (int x = x0; x < x0 + value.dimension(COLS); x++) {
            for (int map = 0; map < value.dimension(MAPS); map++) {
                (*dest)(map, y, x) += value(map, y - y0, x - x0);
            }
        }
    }
}

Vector ApplyReLU(Vector input) {
    for (int i = 0; i < input.size(); i++) {
        input(i) = NN::ReLU(input(i));
    }
    return input;
}

Tensor3D AsTensor3D(const Vector &input, int maps, int cols, int rows) {
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

Matrix AsMatrix(const Tensor4D &input) {
    int f_count = (int) input.dimension(0);
    int maps = (int) input.dimension(MAPS + 1);
    int rows = (int) input.dimension(ROWS + 1);
    int cols = (int) input.dimension(COLS + 1);
    auto res = Matrix(f_count, maps * rows * cols);
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

Tensor4D AsTensor4D(const Matrix &input, int f_count, int maps, int cols, int rows) {
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
