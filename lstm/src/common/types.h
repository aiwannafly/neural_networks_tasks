#ifndef RNN_TYPES_H
#define RNN_TYPES_H

#include "../eigen.h"

namespace NN {
    typedef struct {
        Vector sample;
        Vector expected_output;
    } Example;

    typedef struct RegressionScore {
        float MSE;
        float MAE;
        float RScore;
    } RegressionScore;

    typedef struct ClassificationScore {
        int TP = 0;
        int FP = 0;
        int TN = 0;
        int FN = 0;
    } ClassificationScore;

    typedef struct LSTMParams {
    public:
        Matrix hx;
        Matrix hh;
        Vector biases;

        LSTMParams(size_t input_size, size_t output_size) {
            biases = Vector(output_size);
            hx = Matrix(output_size, input_size);
            hh = Matrix(output_size, output_size);
            biases.setRandom();
            hx.setRandom();
            hh.setRandom();
        }
    } Params;

    typedef struct LSTMDeltas {
    public:
        Matrix W;
        Matrix U;
        Matrix biases;

        LSTMDeltas(size_t input_size, size_t output_size) {
            W = Matrix(output_size * 4, input_size);
            U = Matrix(output_size * 4, output_size);
            biases = Matrix(4, output_size);
            clear();
        }

        void clear() {
            W.setZero();
            U.setZero();
            biases.setZero();
        }
    } LSTMDeltas;
}

#endif //RNN_TYPES_H
