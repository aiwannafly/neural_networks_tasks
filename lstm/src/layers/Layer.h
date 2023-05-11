#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include <cstddef>
#include "../eigen.h"

namespace NN {
    class Layer {
    public:
        virtual ~Layer() = default;

        virtual size_t getInputSize() const = 0;

        virtual size_t getOutputSize() const = 0;

        virtual Vector forward(const Vector& input) = 0;

        virtual Vector backprop(const Vector &input, const Vector& prev_deltas, float l_rate) = 0;
    };
}

#endif //RNN_LAYER_H
