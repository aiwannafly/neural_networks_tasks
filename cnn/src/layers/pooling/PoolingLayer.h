#ifndef CNN_POOLINGLAYER_H
#define CNN_POOLINGLAYER_H


#include "../CNNLayer.h"

namespace cnn {
    class PoolingLayer : public CNNLayer {
    public:
        explicit PoolingLayer(size_t size);

        std::vector<FeatureMap*> *apply(std::vector<FeatureMap*> *maps) override;

        ~PoolingLayer() override = default;

    protected:
        size_t size;

        virtual float getPool(size_t x, size_t y, FeatureMap *map) const = 0;
    };
}


#endif //CNN_POOLINGLAYER_H
