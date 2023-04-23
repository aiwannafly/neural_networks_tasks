#ifndef CNN_POOLINGLAYER_H
#define CNN_POOLINGLAYER_H

#include "../CNNLayer.h"

namespace cnn {
    class PoolingLayer : public CNNLayer {
    public:
        explicit PoolingLayer(int size);

        Tensor3D apply(const Tensor3D &input) override;

        Tensor3D backprop(const Tensor3D &input, const Tensor3D &deltas) override = 0;

        ~PoolingLayer() override = default;

    protected:
        int size;
        std::array<long, 3> extent{};

        virtual float getPool(const Tensor3D &input, const std::array<long, 3> &offset) = 0;
    };
}


#endif //CNN_POOLINGLAYER_H
