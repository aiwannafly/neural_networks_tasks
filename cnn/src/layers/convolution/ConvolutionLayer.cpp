#include <iostream>
#include "ConvolutionLayer.h"
#include "../../utils.h"

namespace cnn {
    ConvolutionLayer::ConvolutionLayer(long coreSize, long coresCount, long inputSlicesCount) {
        this->coreSize = coreSize;
        this->inputMapsCount = inputSlicesCount;
        this->coresCount = coresCount;
        this->cores = new Tensor4D(coresCount, inputSlicesCount, coreSize, coreSize);
        this->cores->setRandom();
        this->biases = new Eigen::VectorXf(coresCount);
        this->biases->setRandom();
    }

    ConvolutionLayer::~ConvolutionLayer() {
        delete cores;
        delete biases;
    }

    long ConvolutionLayer::getCoreSize() const {
        return coreSize;
    }

    Tensor4D *ConvolutionLayer::getCores() {
        return cores;
    }

    float getScalarProd(const Tensor3D &a, const Tensor3D &b) {
        assert(a.dimension(SLICES) == b.dimension(SLICES));
        assert(a.dimension(ROWS) == b.dimension(ROWS));
        assert(a.dimension(COLS) == b.dimension(COLS));
        float res = 0;
        for (int i = 0; i < a.dimension(SLICES); i++) {
            for (int j = 0; j < a.dimension(ROWS); j++) {
                for (int k = 0; k < a.dimension(COLS); k++) {
                    res += a(i, j, k) * b(i, j, k);
                }
            }
        }
        return res;
    }

    Tensor3D ConvolutionLayer::apply(const Tensor3D &input) {
        long edgeOffset = coreSize / 2;
        assert(2 * edgeOffset <= input.dimension(ROWS));
        assert(2 * edgeOffset <= input.dimension(COLS));
        assert(input.dimension(SLICES) == inputMapsCount);

        std::array<long, 3> offset = {0, 0, 0}; // Starting point
        std::array<long, 3> extent{};
        extent[SLICES] = inputMapsCount;
        extent[ROWS] = coreSize;
        extent[COLS] = coreSize;

        std::array<long, 4> coreOffset = {0, 0, 0, 0}; // Starting point
        std::array<long, 4> coreExtent{};
        coreExtent[0] = 1;
        coreExtent[SLICES + 1] = inputMapsCount;
        coreExtent[ROWS + 1] = coreSize;
        coreExtent[COLS + 1] = coreSize;

        long resHeight = MAX(input.dimension(ROWS) - 2 * edgeOffset, 1);
        long resWidth = MAX(input.dimension(COLS) - 2 * edgeOffset, 1);
        Tensor3D result(coresCount, resHeight, resWidth);
//        std::cout << "Result dimensions:" << std::endl;
//        std::cout << result.dimension(SLICES) << std::endl;
//        std::cout << result.dimension(ROWS) << std::endl;
//        std::cout << result.dimension(COLS) << std::endl;

        std::vector<Tensor3D> coresWeights;
        for (long coreId = 0; coreId < coresCount; coreId++) {
            coreOffset[0] = coreId;
            Tensor3D coreWeights = (*cores).slice(coreOffset, coreExtent).reshape(extent);
            coresWeights.push_back(coreWeights);
        }
        Eigen::Tensor<float, 0> scalarProd;
        Tensor3D currentPart;
        for (long y = edgeOffset; y < input.dimension(ROWS) - edgeOffset; y++) {
            offset[ROWS] = y - edgeOffset;
            for (long x = edgeOffset; x < input.dimension(COLS) - edgeOffset; x++) {
                offset[COLS] = x - edgeOffset;
                currentPart = input.slice(offset, extent);
                for (long coreId = 0; coreId < coresCount; coreId++) {
                    float bias = (*biases)(coreId);
                    scalarProd = (currentPart * coresWeights.at(coreId)).sum();
                    float convResult = scalarProd(0) + bias;
                    result(coreId, y - edgeOffset, x - edgeOffset) = convResult;
                }
            }
        }
        return result;
    }

    long ConvolutionLayer::getCoresCount() const {
        return coresCount;
    }

    long ConvolutionLayer::getInputMapsCount() const {
        return inputMapsCount;
    }

    Eigen::VectorXf *ConvolutionLayer::getBiases() {
        return biases;
    }
}
