#ifndef CNN_CNNLAYER_H
#define CNN_CNNLAYER_H

#include "../eigen.h"

namespace cnn {
    class CNNLayer {
    public:
        virtual Tensor3D apply(const Tensor3D &input) = 0;

        /*
         * the layer should use the deltas to correct weights and biases,
         * then it should produce a tensor of its own deltas and return it
         * for the further back propagation
         */
        virtual Tensor3D backprop(const Tensor3D &input, const Tensor3D &deltas) = 0;

        virtual ~CNNLayer() = default;
    };
}

#endif //CNN_CNNLAYER_H
