#include <iostream>
#include "ConvolutionLayer.h"
#include "../../utils.h"

#include "../../common/functions.h"

namespace CNN {
    ConvolutionLayer::ConvolutionLayer(long coreSize, long coresCount, long inputSlicesCount) {
        this->coreSize = coreSize;
        this->inputMapsCount = inputSlicesCount;
        this->coresCount = coresCount;
        this->cores = new Tensor4D(coresCount, inputSlicesCount, coreSize, coreSize);
        this->cores->setRandom();
        this->biases = new Eigen::VectorXf(coresCount);
        this->biases->setRandom();
        extent[SLICES] = inputMapsCount;
        extent[ROWS] = coreSize;
        extent[COLS] = coreSize;
        coreExtent[0] = 1;
        coreExtent[SLICES + 1] = inputMapsCount;
        coreExtent[ROWS + 1] = coreSize;
        coreExtent[COLS + 1] = coreSize;
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

    Tensor3D ConvolutionLayer::apply(const Tensor3D &input) {
        long edgeOffset = coreSize / 2;
        assert(2 * edgeOffset <= input.dimension(ROWS));
        assert(2 * edgeOffset <= input.dimension(COLS));
        assert(input.dimension(SLICES) == inputMapsCount);
        std::array<long, 3> offset = {0, 0, 0};
        std::array<long, 4> coreOffset = {0, 0, 0, 0};

        long resHeight = MAX(input.dimension(ROWS) - 2 * edgeOffset, 1);
        long resWidth = MAX(input.dimension(COLS) - 2 * edgeOffset, 1);
        Tensor3D result(coresCount, resHeight, resWidth);
        std::vector<Tensor3D > coresWeights;
        for (long coreId = 0; coreId < coresCount; coreId++) {
            coreOffset[0] = coreId;
            Tensor3D coreWeights = (*cores).slice(coreOffset, coreExtent).reshape(extent);
            coresWeights.push_back(coreWeights);
        }
        Eigen::Tensor<float, 0> scalarProd;
        Tensor3D currentPart;
        long evenOffset = 0;
        if (coreSize % 2 == 0) {
            evenOffset = 1;
        }
        for (long y = edgeOffset; y < input.dimension(ROWS) - edgeOffset + evenOffset; y++) {
            offset[ROWS] = y - edgeOffset;
            for (long x = edgeOffset; x < input.dimension(COLS) - edgeOffset + evenOffset; x++) {
                offset[COLS] = x - edgeOffset;
                currentPart = input.slice(offset, extent);
                for (long coreId = 0; coreId < coresCount; coreId++) {
                    float bias = (*biases)(coreId);
                    scalarProd = (currentPart * coresWeights.at(coreId)).sum();
                    result(coreId, y - edgeOffset, x - edgeOffset) = ReLU(scalarProd(0) + bias);
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

    Tensor3D ConvolutionLayer::backprop(const Tensor3D &input, const Tensor3D &deltas, float learningRate) {
        Tensor3D newDeltas = Tensor3D(input.dimension(SLICES), input.dimension(ROWS), input.dimension(COLS));
        newDeltas.setZero();
        // here we need to make a **full convolution** between 180 degrees rotated filters and deltas
        // idea: https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c#6042
        Tensor4D coresRotated = getRotatedCores();
        long expansion = (long) deltas.dimension(ROWS) - 1;
        Tensor4D extendedCores = Tensor4D(coresCount, inputMapsCount, coreSize + 2 * expansion,
                                          coreSize + 2 * expansion);
        extendedCores.setZero();
        for (int coreId = 0; coreId < coresCount; coreId++) {
            for (int mapId = 0; mapId < inputMapsCount; mapId++) {
                for (int i = 0; i < coreSize; i++) {
                    for (int j = 0; j < coreSize; j++) {
                        extendedCores(coreId, mapId, i + expansion, j + expansion) = coresRotated(coreId,
                                                                                                  mapId, i, j);
                    }
                }
            }
        }
        long backPropCoreSize = (long) deltas.dimension(ROWS);
        long edgeOffset = backPropCoreSize / 2;
        long evenOffset = 0;
        if (backPropCoreSize % 2 == 0) {
            evenOffset = 1;
        }
        std::array<long, 3> partExtent{};
        partExtent[SLICES] = 1;
        partExtent[ROWS] = backPropCoreSize;
        partExtent[COLS] = backPropCoreSize;
        std::array<long, 3> coresOffset{0, 0, 0};
        std::array<long, 3> deltasOffset{0, 0, 0};
        for (long y = edgeOffset; y < coreSize - edgeOffset + evenOffset; y++) {
            coresOffset[ROWS] = y - edgeOffset;
            for (long x = edgeOffset; x < coreSize - edgeOffset + evenOffset; x++) {
                coresOffset[COLS] = x - edgeOffset;
                for (long coreId = 0; coreId < coresCount; coreId++) {
                    deltasOffset[SLICES] = coreId;
                    Tensor3D deltasOfCore = deltas.slice(deltasOffset, partExtent);
                    for (long mapId = 0; mapId < inputMapsCount; mapId++) {
                        coresOffset[SLICES] = mapId;
                        Tensor3D coresPart = input.slice(coresOffset, partExtent);
                        Eigen::Tensor<float, 0> scalarProd = (deltasOfCore * coresPart).sum();
                        newDeltas(mapId, y - edgeOffset, x - edgeOffset) += scalarProd(0);
                    }
                }
            }
        }
        changeWeights(input, deltas, learningRate);
        return newDeltas;
    }

    void ConvolutionLayer::changeWeights(const Tensor3D &input, const Tensor3D &deltas, float learningRate) {
        long backPropCoreSize = (long) deltas.dimension(ROWS);
        long edgeOffset = backPropCoreSize / 2;
        long evenOffset = 0;
        if (backPropCoreSize % 2 == 0) {
            evenOffset = 1;
        }
        // for the back propagation we will use convolution again, this time with use of the output deltas
        // idea: https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c#6042
        std::array<long, 3> partExtent{};
        partExtent[SLICES] = 1;
        partExtent[ROWS] = backPropCoreSize;
        partExtent[COLS] = backPropCoreSize;
        std::array<long, 3> inputOffset{0, 0, 0};
        std::array<long, 3> deltasOffset{0, 0, 0};
        for (long y = edgeOffset; y < input.dimension(ROWS) - edgeOffset + evenOffset; y++) {
            inputOffset[ROWS] = y - edgeOffset;
            for (long x = edgeOffset; x < input.dimension(COLS) - edgeOffset + evenOffset; x++) {
                inputOffset[COLS] = x - edgeOffset;
                for (long coreId = 0; coreId < coresCount; coreId++) {
                    deltasOffset[SLICES] = coreId;
                    Tensor3D deltasOfCore = deltas.slice(deltasOffset, partExtent);
                    for (long mapId = 0; mapId < inputMapsCount; mapId++) {
                        inputOffset[SLICES] = mapId;
                        Tensor3D inputMapPart = input.slice(inputOffset, partExtent);
                        Eigen::Tensor<float, 0> scalarProd = (deltasOfCore * inputMapPart).sum();
                        (*cores)(coreId, mapId, y - edgeOffset, x - edgeOffset) += scalarProd(0) * learningRate;
                    }
                    Eigen::Tensor<float, 0> deltasSum = deltasOfCore.sum();
                    (*biases)(coreId) += deltasSum(0) * learningRate;
                }
            }
        }
    }

    Tensor4D ConvolutionLayer::getRotatedCores() const {
        Tensor4D coresCopy = *cores;
        for (int coreId = 0; coreId < coresCount; coreId++) {
            for (int mapId = 0; mapId < inputMapsCount; mapId++) {
                for (int i = 0; i < coreSize / 2; i++) {
                    for (int j = 0; j < coreSize; j++) {
                        float temp = coresCopy(coreId, mapId, i, j);
                        coresCopy(coreId, mapId, i, j) = coresCopy(coreId, mapId, coreSize - i - 1, j);
                        coresCopy(coreId, mapId, coreSize - i - 1, j) = temp;
                    }
                }
                for (int i = 0; i < coreSize / 2; i++) {
                    for (int j = 0; j < coreSize; j++) {
                        float temp = coresCopy(coreId, mapId, j, i);
                        coresCopy(coreId, mapId, j, i) = coresCopy(coreId, mapId, j, coreSize - i - 1);
                        coresCopy(coreId, mapId, j, coreSize - i - 1) = temp;
                    }
                }
            }
        }
        return coresCopy;
    }
}
