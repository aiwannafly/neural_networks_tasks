//
// Created by ms_dr on 20.04.2023.
//

#ifndef CNN_FEATUREMAP_H
#define CNN_FEATUREMAP_H


#include <cstddef>

namespace cnn {
    class FeatureMap {
    public:
        virtual size_t getWidth() = 0;

        virtual size_t getHeight() = 0;

        virtual float getValue(size_t x, size_t y) = 0;

        virtual void setValue(size_t x, size_t y, float value) = 0;

        virtual ~FeatureMap() = default;
    };
}

#endif //CNN_FEATUREMAP_H
