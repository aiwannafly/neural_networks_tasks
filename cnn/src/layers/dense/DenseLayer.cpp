#include "DenseLayer.h"

#include <cmath>
#include <iostream>

#include "../../common/functions.h"

namespace perceptron {

    DenseLayer::DenseLayer(size_t prev_layer_size, size_t current_layer_size) {
        this->output_size = current_layer_size;
        this->input_size = prev_layer_size;
        biases = new Vector(current_layer_size);
        weights = new  Matrix(current_layer_size, prev_layer_size);
//        biases->setZero();
//        weights->setZero();
        biases->setZero();
        weights->setZero();
    }

    DenseLayer::~DenseLayer() {
        delete biases;
        delete weights;
    }

    Vector DenseLayer::forward(const Vector& input) {
        Vector sum = *weights * input + *biases;
        for (int i = 0; i < sum.size(); i++) {
            sum(i) = Tanh(sum(i));
        }
        // with use of vector op
        return sum;
    }

    Vector DenseLayer::act_deriv(const Vector &output) {
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
//            (*biases)(j) += common_err;
        }
        return next_deltas;
    }

    Matrix DenseLayer::weightsTemplate() const {
        Matrix t = *weights;
        t.setZero();
        return t;
    }

    void DenseLayer::applyWightsDeltas(const Matrix &w_deltas) {
        (*weights) += w_deltas;
    }
}
