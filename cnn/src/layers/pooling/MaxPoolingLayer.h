#ifndef CNN_MAXPOOLINGLAYER_H
#define CNN_MAXPOOLINGLAYER_H

#include "PoolingLayer.h"
#include "../../utils.h"

namespace CNN {
    class MaxPoolingLayer : public PoolingLayer {
    public:
        explicit MaxPoolingLayer(int size) : PoolingLayer(size) {}

        Tensor3D backprop(const Tensor3D &input, const Tensor3D &deltas, float learningRate) override;

        Tensor3D apply(const Tensor3D &input) override;

    private:
        LongsTensor3D indicators;

        float getPool(const Tensor3D &input, const std::array<long, 3> &offset) override;
    };
}

#endif //CNN_MAXPOOLINGLAYER_H
