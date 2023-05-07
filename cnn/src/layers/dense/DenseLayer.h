#ifndef PERCEPTRON_LAYER_H
#define PERCEPTRON_LAYER_H

#include "../../eigen.h"

namespace perceptron {
    class DenseLayer {
    public:
        DenseLayer(size_t prev_layer_size, size_t current_layer_size);

        ~DenseLayer();

        size_t getInputSize() const;

        size_t getOutputSize() const;

        // returns activation(W * input + bias)
        Vector forward(const Vector& input);

        Vector backprop(const Vector &input, const Vector& prev_deltas, float l_rate);

        Matrix weightsTemplate() const;

        void applyWightsDeltas(const Matrix &w_deltas);

        static Vector act_deriv(const Vector &output);

    private:
        size_t input_size;
        size_t output_size;
        Matrix *weights;
        Vector *biases;
    };
}

#endif //PERCEPTRON_LAYER_H
