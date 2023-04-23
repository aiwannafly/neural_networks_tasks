#ifndef CNN_MAXPOOLINGLAYER_H
#define CNN_MAXPOOLINGLAYER_H

#include "PoolingLayer.h"
#include "../../utils.h"

namespace cnn {
    class MaxPoolingLayer : public PoolingLayer {
    public:
        explicit MaxPoolingLayer(int size) : PoolingLayer(size) {}

    private:
        float getPool(Tensor3D slicePart) const override {
            Eigen::Tensor<float, 0> maximum = slicePart.maximum();
            return maximum(0);
        }
    };
}

#endif //CNN_MAXPOOLINGLAYER_H
