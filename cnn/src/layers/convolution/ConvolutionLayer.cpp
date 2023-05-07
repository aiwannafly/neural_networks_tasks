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
                        (*filters)(i, j, k, t) -= 0.5;
//                        (*filters)(i, j, k, t) *= 2;
                        //(*filters)(i, j, k, t) -= 1;
                    }
                }
            }
        }
        this->biases = new Vector(f_cnt);
        this->biases->setZero();
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

    Tensor3D ConvolutionLayer::forward(const Tensor3D &input) {
        assert(f_size <= input.dimension(ROWS));
        assert(f_size <= input.dimension(COLS));
        long edge = f_size / 2;
        assert(input.dimension(MAPS) == input_maps);
        std::array<long, 3> offset = {0, 0, 0};

        long outHeight = MAX(input.dimension(ROWS) - 2 * edge, 1);
        long outWidth = MAX(input.dimension(COLS) - 2 * edge, 1);
        Tensor3D output(f_cnt, outHeight, outWidth);
        Matrix weights = AsMatrix(*filters);
        int y = 0;
        while (y + f_size <= input.dimension(ROWS)) {
            offset[ROWS] = y;
            int x = 0;
            while (x + f_size <= input.dimension(COLS)) {
                offset[COLS] = x;
                Vector inputVector = AsVector(input.slice(offset, extent).reshape(extent));
                Vector outputVector = ApplyReLU(weights * inputVector + (*biases));
                SetTensor3DValue(&output, x, y, outputVector);
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
        Matrix weights = AsMatrix(*filters);
        int y = 0;
        int input_slice_size = input_maps * f_size * f_size;

//        LOG("\nWeights before:");
//        PrintMatrix(weights);

//        LOG("\nPrev deltas:");
//        PrintTensor3D(output_deltas);

        while (y + f_size <= input.dimension(ROWS)) {
            offset[ROWS] = y;
            int x = 0;
            while (x + f_size <= input.dimension(COLS)) {
                offset[COLS] = x;
                Vector prev_deltas = GetTensor3DValue(output_deltas, x, y);
                Vector input_slice = AsVector(input.slice(offset, extent).reshape(extent));

                // scoring next deltas
                Vector input_act_deriv = ReLUDeriv(input_slice);
                Vector curr_deltas = Vector(input_slice_size);
                for (int i = 0; i < input_slice_size; i++) {
                    float scalar_prod = weights.col(i).dot(prev_deltas);
                    curr_deltas(i) = scalar_prod * input_act_deriv(i);
                }
                Tensor3D tensorDeltas = AsTensor3D(curr_deltas, input_maps, f_size, f_size);
                AddTensorPart(&nextDeltas, x, y, tensorDeltas);

                // scoring filter deltas
                for (int j = 0; j < f_cnt; j++) {
                    float common_err = l_rate * prev_deltas(j);
                    for (int k = 0; k < input_slice_size; k++) {
                        float err = common_err * input_slice(k);
                        weights(j, k) += err;
                    }
//                    (*biases)(j) += common_err;
                }
                x++;
            }
            y++;
        }
//        LOG("\nWeights after:");
//        PrintMatrix(weights);

//        LOG("Next deltas:");
//        PrintTensor3D(nextDeltas);

        *filters = AsTensor4D(weights, f_cnt, input_maps, f_size, f_size);
        return nextDeltas;
    }

    long ConvolutionLayer::getFiltersCount() const {
        return f_cnt;
    }

    long ConvolutionLayer::getInputMapsCount() const {
        return input_maps;
    }
}
