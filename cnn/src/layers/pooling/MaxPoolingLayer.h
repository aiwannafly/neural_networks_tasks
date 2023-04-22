//
// Created by ms_dr on 20.04.2023.
//

#ifndef CNN_MAXPOOLINGLAYER_H
#define CNN_MAXPOOLINGLAYER_H


#include "PoolingLayer.h"
#include "../../utils.h"

namespace cnn {
    class MaxPoolingLayer : public PoolingLayer {
    public:
        explicit MaxPoolingLayer(size_t size) : PoolingLayer(size) {}

    private:
        float getPool(size_t x, size_t y, FeatureMap *map) const override {
            float pooled = map->getValue(x, y);
            for (size_t i = y; i < size; i++) {
                for (size_t j = x; j < size; j++) {
                    pooled = utils::Maxf(pooled, map->getValue(j, i));
                }
            }
            return pooled;
        }
    };
}


#endif //CNN_MAXPOOLINGLAYER_H
