//
// Created by ms_dr on 20.04.2023.
//

#include "VectorFeatureMap.h"

#include <utility>

cnn::VectorFeatureMap::VectorFeatureMap(size_t width, size_t height) {
    this->width = width;
    this->height = height;
    map = Eigen::VectorXf(width * height);
}

cnn::VectorFeatureMap::VectorFeatureMap(size_t width, size_t height, Eigen::VectorXf base) {
    this->width = width;
    this->height = height;
    assert(base.size() == width * height);
    this->map = std::move(base);
}

float cnn::VectorFeatureMap::getValue(size_t x, size_t y) {
    size_t pos = x + y * width;
    assert(pos < map.size());
    assert(pos >= 0);
    return map((int) pos);
}

void cnn::VectorFeatureMap::setValue(size_t x, size_t y, float value) {
    size_t pos = x + y * width;
    assert(pos < map.size());
    assert(pos >= 0);
    map((int) pos) = value;
}

size_t cnn::VectorFeatureMap::getWidth() {
    return width;
}

size_t cnn::VectorFeatureMap::getHeight() {
    return height;
}

Eigen::VectorXf cnn::VectorFeatureMap::asVector() {
    return map;
}
