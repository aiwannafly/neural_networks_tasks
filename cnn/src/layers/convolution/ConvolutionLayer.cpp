#include "ConvolutionLayer.h"
#include "../../utils.h"
#include "../../feature_map/VectorFeatureMap.h"

namespace cnn {
    ConvolutionLayer::ConvolutionLayer(size_t coreSize, size_t outputMapsCount, size_t inputMapsCount) {
        this->coreSize = coreSize;
        this->inputMapsCount = inputMapsCount;
        this->cores = new std::vector<Eigen::MatrixXf *>;
        for (int i = 0; i < outputMapsCount; i++) {
            auto *core = new Eigen::MatrixXf(coreSize, coreSize);
            core->setRandom();
            cores->push_back(core);
        }
    }

    ConvolutionLayer::~ConvolutionLayer() {
        for (auto *core: *cores) {
            delete core;
        }
        delete cores;
    }

    size_t ConvolutionLayer::getCoreSize() const {
        return coreSize;
    }

    std::vector<Eigen::MatrixXf *> *ConvolutionLayer::getCores() {
        return cores;
    }

    std::vector<FeatureMap *> *ConvolutionLayer::apply(std::vector<FeatureMap *> *maps) {
        auto *result = new std::vector<FeatureMap *>;
        for (auto *map: *maps) {
            for (auto *core: *cores) {
                result->push_back(applyCore(map, core));
            }
        }
        return result;
    }

    FeatureMap *ConvolutionLayer::applyCore(FeatureMap *map, Eigen::MatrixXf *core) const {
        size_t offset = coreSize / 2;
        assert(2 * offset <= map->getHeight());
        assert(2 * offset <= map->getWidth());
        size_t resultedWidth = utils::MaxZ(map->getWidth() - 2 * offset, 1);
        size_t resultedHeight = utils::MaxZ(map->getHeight() - 2 * offset, 1);
        FeatureMap *resultedMap = new VectorFeatureMap(resultedWidth, resultedHeight);
        for (size_t y = offset; y < map->getHeight() - offset; y++) {
            for (size_t x = offset; x < map->getWidth() - offset; x++) {
                float res = 0;
                for (size_t i = y - offset; i < coreSize; i++) {
                    for (size_t j = x - offset; j < coreSize; j++) {
                        size_t mX = j - x + offset;
                        size_t mY = i - y + offset;
                        res += map->getValue(i, j) * (*core)((int) mY, (int) mX);
                    }
                }
                resultedMap->setValue(x - offset, y - offset, res);
            }
        }
        return resultedMap;
    }

    size_t ConvolutionLayer::getOutputMapsCount() const {
        return cores->size();
    }

    size_t ConvolutionLayer::getInputMapsCount() const {
        return inputMapsCount;
    }
}
