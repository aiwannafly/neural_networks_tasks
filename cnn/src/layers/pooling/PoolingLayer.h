#ifndef CNN_POOLINGLAYER_H
#define CNN_POOLINGLAYER_H

#include "../CNNLayer.h"

namespace cnn {
    class PoolingLayer : public CNNLayer {
    public:
        explicit PoolingLayer(int size);

        Tensor3D apply(const Tensor3D &input) override;

        ~PoolingLayer() override = default;

    protected:
        int size;

        virtual float getPool(Tensor3D slicePart) const = 0;
    };
}


#endif //CNN_POOLINGLAYER_H
