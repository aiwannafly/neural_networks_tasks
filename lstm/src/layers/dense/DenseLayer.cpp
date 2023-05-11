#include "DenseLayer.h"

#include <cmath>

#include "../../common/functions.h"

namespace NN {

    DenseLayer::DenseLayer(size_t input_size, size_t output_size) {
        this->output_size = output_size;
        this->input_size = input_size;
        biases = new Vector(output_size);
        weights = new Matrix(output_size, input_size);
        biases->setZero();
        weights->setZero();
    }

    DenseLayer::~DenseLayer() {
        delete biases;
        delete weights;
    }

    Vector DenseLayer::forward(const Vector& input) {
        return TanhV(*weights * input + *biases);
    }

    Vector DenseLayer::actDeriv(const Vector &output) {
        return TanhDeriv(output);
    }

    size_t DenseLayer::getInputSize() const {
        return input_size;
    }

    size_t DenseLayer::getOutputSize() const {
        return output_size;
    }

    Vector DenseLayer::backprop(const Vector &input, const Vector &prev_deltas, float l_rate) {
        auto next_deltas = Vector(input_size);
        auto activ_deriv = SigmoidDeriv(input);
        for (int i = 0; i < input_size; i++) {
            float scalar_prod = weights->col(i).dot(prev_deltas);
            next_deltas(i) = scalar_prod * activ_deriv(i);
        }

        for (int j = 0; j < output_size; j++) {
            float common_err = l_rate * prev_deltas(j);
            for (int i = 0; i < input_size; i++) {
                (*weights)(j, i) += common_err * input(i);
            }
            (*biases)(j) += common_err;
        }
        return next_deltas;
    }
}
