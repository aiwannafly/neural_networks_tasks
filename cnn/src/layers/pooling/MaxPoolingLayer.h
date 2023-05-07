#ifndef CNN_MAXPOOLINGLAYER_H
#define CNN_MAXPOOLINGLAYER_H

#include "PoolingLayer.h"
#include "../../utils/utils.h"

namespace CNN {
    class MaxPoolingLayer : public PoolingLayer {
    public:
        explicit MaxPoolingLayer(int size) : PoolingLayer(size) {}

        Tensor3D backprop(const Tensor3D &input, const Tensor3D &deltas, float l_rate) override;

        Tensor3D forward(const Tensor3D &input) override;

    private:
        LongsTensor3D rowIndicators;
        LongsTensor3D colIndicators;

        float getPool(const Tensor3D &input, const std::array<long, 3> &offset) override;
    };
}

#endif //CNN_MAXPOOLINGLAYER_H
