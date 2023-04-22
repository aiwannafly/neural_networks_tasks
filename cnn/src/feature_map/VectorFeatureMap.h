//
// Created by ms_dr on 20.04.2023.
//

#ifndef CNN_VECTORFEATUREMAP_H
#define CNN_VECTORFEATUREMAP_H

#include "FeatureMap.h"

#include "../eigen.h"

namespace cnn {
    class VectorFeatureMap : public FeatureMap {
    public:
        VectorFeatureMap(size_t width, size_t height);

        VectorFeatureMap(size_t width, size_t height, Eigen::VectorXf base);

        float getValue(size_t x, size_t y) override;

        void setValue(size_t x, size_t y, float value) override;

        size_t getWidth() override;

        size_t getHeight() override;

        Eigen::VectorXf asVector();

        ~VectorFeatureMap() override = default;
    private:
        size_t width;
        size_t height;
        Eigen::VectorXf map;
    };
}

#endif //CNN_VECTORFEATUREMAP_H
