#include <iostream>
#include "ConvolutionLayer.h"
#include "../../utils/utils.h"

#include "../../common/functions.h"

namespace CNN {
    ConvolutionLayer::ConvolutionLayer(long f_size, long f_cnt, long input_maps) {
        this->f_size = f_size;
        this->input_maps = input_maps;
        this->f_cnt = f_cnt;
        this->filters = new Tensor4D(f_cnt, input_maps, f_size, f_size);
        this->filters->setRandom();
        for (int i = 0; i < f_cnt; i++) {
            for (int j = 0; j < input_maps; j++) {
                for (int k = 0; k < f_size; k++) {
                    for (int t = 0; t < f_size; t++) {
                        (*filters)(i, j, k, t) *= 2;
                        (*filters)(i, j, k, t) -= 1;
                    }
                }
            }
        }
        this->biases = new Eigen::VectorXf(f_cnt);
        this->biases->setRandom();
        extent[MAPS] = input_maps;
        extent[ROWS] = f_size;
        extent[COLS] = f_size;
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

    Tensor3D ConvolutionLayer::forward(const Tensor3D &input) {
        long edge = f_size / 2;
        assert(2 * edge <= input.dimension(ROWS));
        assert(2 * edge <= input.dimension(COLS));
        assert(input.dimension(MAPS) == input_maps);
        std::array<long, 3> offset = {0, 0, 0};

        long outHeight = MAX(input.dimension(ROWS) - 2 * edge, 1);
        long outWidth = MAX(input.dimension(COLS) - 2 * edge, 1);
        Tensor3D output(f_cnt, outHeight, outWidth);
        Matrix weights = toMatrix(*filters);
        int y = 0;
        while (y + f_size <= input.dimension(ROWS)) {
            offset[ROWS] = y;
            int x = 0;
            while (x + f_size <= input.dimension(COLS)) {
                offset[COLS] = x;
                Vector inputVector = toVector(input.slice(offset, extent).reshape(extent));
                Vector outputVector = applyReLU(weights * inputVector + (*biases));
                setTensor3DValue(&output, x, y, outputVector);
                x++;
            }
            y++;
        }
        return output;
    }

    Tensor3D ConvolutionLayer::backprop(const Tensor3D &input, const Tensor3D &output_deltas, float l_rate) {
        std::array<long, 3> offset = {0, 0, 0};
        Tensor3D nextDeltas = Tensor3D(input.dimension(MAPS), input.dimension(ROWS), input.dimension(COLS));
        nextDeltas.setZero();
        Matrix weights = toMatrix(*filters);
        int y = 0;
        int slice_size = input_maps * f_size * f_size;
        while (y + f_size <= input.dimension(ROWS)) {
            offset[ROWS] = y;
            int x = 0;
            while (x + f_size <= input.dimension(COLS)) {
                offset[COLS] = x;
                Vector prev_deltas = getTensor3DValue(output_deltas, x, y);
                Vector input_slice = toVector(input.slice(offset, extent).reshape(extent));

                // scoring next deltas
                Vector input_act_deriv = ReLU_deriv(input_slice);
                Vector curr_deltas = Vector(slice_size);
                for (int i = 0; i < slice_size; i++) {
                    Vector weights_col = weights.col(i);
                    float scalar_prod = weights_col.dot(prev_deltas);
                    curr_deltas(i) = scalar_prod * input_act_deriv(i);
                }
                Tensor3D tensorDeltas = toTensor3D(curr_deltas, input_maps, f_size, f_size);
                addTensorPart(&nextDeltas, x, y, tensorDeltas);

                // scoring filter deltas
                for (int j = 0; j < f_cnt; j++) {
                    float common_err = l_rate * prev_deltas(j);
                    for (int k = 0; k < slice_size; k++) {
                        float err = common_err * input_slice(k);
                        weights(j, k) += err;
                    }
                    (*biases)(j) += common_err;
                }
                x++;
            }
            y++;
        }
        *filters = toTensor4D(weights, f_cnt, input_maps, f_size, f_size);
        return nextDeltas;
    }

    long ConvolutionLayer::getFiltersCount() const {
        return f_cnt;
    }

    long ConvolutionLayer::getInputMapsCount() const {
        return input_maps;
    }
}
