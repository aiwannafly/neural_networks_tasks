#ifndef CNN_POOLINGLAYER_H
#define CNN_POOLINGLAYER_H

#include "../CNNLayer.h"

namespace CNN {
    class PoolingLayer : public CNNLayer {
    public:
        explicit PoolingLayer(int size);

        Tensor3D forward(const Tensor3D &input) override;

        Tensor3D backprop(const Tensor3D &input, const Tensor3D &deltas, float learningRate) override = 0;

        Matrix weightsTemplate() const override;

        void applyWeightsDeltas(const Matrix &w_deltas) override {};

        ~PoolingLayer() override = default;

    protected:
        int size;
        std::array<long, 3> extent{};

        virtual float getPool(const Tensor3D &input, const std::array<long, 3> &offset) = 0;
    };
}


#endif //CNN_POOLINGLAYER_H
