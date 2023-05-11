#ifndef RNN_TYPES_H
#define RNN_TYPES_H

#include "../eigen.h"

namespace NN {
    typedef struct {
        Tensor3D sample;
        Vector expected_output;
    } Example;

    typedef struct ClassificationScore {
        int TP = 0;
        int FP = 0;
        int TN = 0;
        int FN = 0;
    } ClassificationScore;

    typedef struct Params {
    public:
        Matrix weights;
        Vector biases;

        Params(size_t input_size, size_t output_size) {
            biases = Vector(output_size);
            weights = Matrix(output_size, input_size);
            biases.setZero();
            weights.setZero();
        }
    } Params;
}

#endif //RNN_TYPES_H
