#include <iostream>
#include "ConvolutionLayer.h"
#include "../../utils/utils.h"

#include "../../common/functions.h"

namespace CNN {
    ConvolutionLayer::ConvolutionLayer(long coreSize, long coresCount, long inputSlicesCount) {
        this->f_size = coreSize;
        this->inputMapsCount = inputSlicesCount;
        this->f_cnt = coresCount;
        this->filters = new Tensor4D(coresCount, inputSlicesCount, coreSize, coreSize);
        this->filters->setRandom();
        for (int i = 0; i < coresCount; i++) {
            for (int j = 0; j < inputSlicesCount; j++) {
                for (int k = 0; k < coreSize; k++) {
                    for (int t = 0; t < coreSize; t++) {
                        (*filters)(i, j, k, t) *= 2;
                        (*filters)(i, j, k, t) -= 1;
//                        (*filters)(i, j, k, t) -= 0.5;
//                        (*filters)(i, j, k, t) *= 2;
                    }
                }
            }
        }
        this->biases = new Eigen::VectorXf(coresCount);
        this->biases->setRandom();
        extent[MAPS] = inputMapsCount;
        extent[ROWS] = coreSize;
        extent[COLS] = coreSize;
        f_extent[0] = 1;
        f_extent[MAPS + 1] = inputMapsCount;
        f_extent[ROWS + 1] = coreSize;
        f_extent[COLS + 1] = coreSize;
    }

    ConvolutionLayer::~ConvolutionLayer() {
        delete filters;
        delete biases;
    }

    long ConvolutionLayer::getCoreSize() const {
        return f_size;
    }

    Tensor4D *ConvolutionLayer::getCores() {
        return filters;
    }

    Tensor3D ConvolutionLayer::apply(const Tensor3D &input) {
        long edge = f_size / 2;
        assert(2 * edge <= input.dimension(ROWS));
        assert(2 * edge <= input.dimension(COLS));
        assert(input.dimension(MAPS) == inputMapsCount);
        std::array<long, 3> offset = {0, 0, 0};
        std::array<long, 4> f_offset = {0, 0, 0, 0};

        long outHeight = MAX(input.dimension(ROWS) - 2 * edge, 1);
        long outWidth = MAX(input.dimension(COLS) - 2 * edge, 1);
        Tensor3D output(f_cnt, outHeight, outWidth);
        long even = 0;
        if (f_size % 2 == 0) {
            even = 1;
        }
        for (long curr_f = 0; curr_f < f_cnt; curr_f++) {
            float bias = (*biases)(curr_f);
            f_offset[0] = curr_f;
            Tensor3D filter = (*filters).slice(f_offset, f_extent).reshape(extent);
            for (long y = edge; y < input.dimension(ROWS) - edge + even; y++) {
                offset[ROWS] = y - edge;
                for (long x = edge; x < input.dimension(COLS) - edge + even; x++) {
                    offset[COLS] = x - edge;
                    Tensor3D currentPart = input.slice(offset, extent);
                    Eigen::Tensor<float, 0> scalarProd = (currentPart * filter).sum();
                    output(curr_f, y - edge, x - edge) = ReLU(scalarProd(0) + bias);
                }
            }
        }
        return output;
    }

    long ConvolutionLayer::getFiltersCount() const {
        return f_cnt;
    }

    long ConvolutionLayer::getInputMapsCount() const {
        return inputMapsCount;
    }

    Tensor3D ConvolutionLayer::backprop(const Tensor3D &input, const Tensor3D &deltas, float l_rate) {
        std::array<long, 3> offset = {0, 0, 0};
        std::array<long, 4> f_offset = {0, 0, 0, 0};
        Tensor3D nextDeltas = Tensor3D(input.dimension(MAPS), input.dimension(ROWS), input.dimension(COLS));
        nextDeltas.setZero();
        for (int curr_f = 0; curr_f < f_cnt; curr_f++) {
            f_offset[0] = curr_f;
            Tensor3D filter = (*filters).slice(f_offset, f_extent).reshape(extent);
            int curr_y = 0;
            int out_y = 0;
            int curr_x = 0;
            int out_x = 0;
            while (curr_y + f_size <= input.dimension(ROWS)) {
                offset[ROWS] = curr_y;
                while (curr_x + f_size <= input.dimension(COLS)) {
                    offset[COLS] = curr_x;
                    float delta = deltas(curr_f, out_y, out_x);
                    for (int i = 0; i < input.dimension(MAPS); i++) {
                        for (int j = 0; j < f_size; j++) {
                            for (int k = 0; k < f_size; k++) {
                                (*filters)(curr_f, i, j, k) += delta * input(i, curr_y + j, curr_x + k) * l_rate;
                            }
                        }
                    }
                    for (int i = 0; i < input.dimension(MAPS); i++) {
                        for (int j = 0; j < f_size; j++) {
                            for (int k = 0; k < f_size; k++) {
                                nextDeltas(i, curr_y + j, curr_x + k) += delta * filter(i, j, k);
                            }
                        }
                    }
                    curr_x += 1;
                    out_x += 1;
                }
                curr_y += 1;
                out_y += 1;
            }
            for (int i = 0; i < deltas.dimension(ROWS); i++) {
                for (int j = 0; j < deltas.dimension(COLS); j++) {
                    (*biases)(curr_f) += deltas(curr_f, i, j);
                }
            }
        }
        return nextDeltas;
    }
}
