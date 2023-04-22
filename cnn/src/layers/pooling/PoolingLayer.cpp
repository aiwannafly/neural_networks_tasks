#include "PoolingLayer.h"

#include "../../utils.h"
#include "../../feature_map/VectorFeatureMap.h"

namespace cnn{
    PoolingLayer::PoolingLayer(size_t size) {
        assert(size != 0);
        this->size = size;
    }

    std::vector<FeatureMap*> *PoolingLayer::apply(std::vector<FeatureMap*> *maps) {
        auto *result = new std::vector<FeatureMap*>();
        for (auto *map : *maps) {
            assert(map->getHeight() % size == 0);
            assert(map->getWidth() % size == 0);
            FeatureMap *pooledMap = new VectorFeatureMap(map->getWidth() / size, map->getHeight() / size);
            for (size_t y = 0; y < map->getHeight(); y += size) {
                for (size_t x = 0; x < map->getWidth(); x += size) {
                    pooledMap->setValue(x, y, getPool(x, y, map));
                }
            }
            result->push_back(pooledMap);
        }
        return result;
    }
}
