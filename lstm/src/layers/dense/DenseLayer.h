#ifndef PERCEPTRON_LAYER_H
#define PERCEPTRON_LAYER_H

#include "../../eigen.h"
#include "../Layer.h"

namespace NN {
    class DenseLayer : public Layer{
    public:
        DenseLayer(size_t input_size, size_t output_size);

        ~DenseLayer() override;

        size_t getInputSize() const override;

        size_t getOutputSize() const override;

        Vector forward(const Vector& input) override;

        Vector backprop(const Vector &input, const Vector& prev_deltas, float l_rate) override;

        static Vector actDeriv(const Vector &output);

    private:
        size_t input_size;
        size_t output_size;
        Matrix *weights;
        Vector *biases;
    };
}

#endif //PERCEPTRON_LAYER_H
