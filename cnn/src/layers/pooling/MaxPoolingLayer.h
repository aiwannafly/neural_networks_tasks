#ifndef CNN_MAXPOOLINGLAYER_H
#define CNN_MAXPOOLINGLAYER_H

#include "PoolingLayer.h"
#include "../../utils.h"

namespace cnn {
    class MaxPoolingLayer : public PoolingLayer {
    public:
        explicit MaxPoolingLayer(int size) : PoolingLayer(size) {}

        Tensor3D backprop(const Tensor3D &input, const Tensor3D &deltas) override;

        Tensor3D apply(const Tensor3D &input) override;

    private:
        LongsTensor3D indicators;

        float getPool(const Tensor3D &input, const std::array<long, 3> &offset) override {
            long slice = offset[SLICES];
            int rowMax = offset[ROWS];
            int colMax = offset[COLS];
            float max = input(slice, rowMax, colMax);
            for (int row = offset[ROWS]; row < offset[ROWS] + size; row++) {
                for (int col = offset[COLS]; col < offset[COLS] + size; col++) {
                    if (input(slice, row, col) > max) {
                        max = input(slice, row, col);
                        rowMax = row;
                        colMax = col;
                    }
                }
            }
            indicators(slice, offset[ROWS] / size, offset[COLS] / size) = colMax + rowMax * (int) input.dimension(COLS);
            return max;
        }
    };

    Tensor3D MaxPoolingLayer::backprop(const Tensor3D &input, const Tensor3D &deltas) {
        Tensor3D new_deltas = Tensor3D(input.dimension(SLICES), input.dimension(ROWS), input.dimension(COLS));
        new_deltas.setZero();
        for (int slice = 0; slice < indicators.dimension(SLICES); slice++) {
            for (int i = 0; i < indicators.dimension(ROWS); i++) {
                for (int j = 0; j < indicators.dimension(COLS); j++) {
                    int pos = indicators(slice, i, j);
                    int col = pos % (int) input.dimension(COLS);
                    int row = (pos - col) / (int) input.dimension(COLS);
                    new_deltas(slice, row, col) = deltas(slice, i, j);
                }
            }
        }
        return new_deltas;
    }

    Tensor3D MaxPoolingLayer::apply(const Eigen::Tensor<float, 3> &input) {
        assert(input.dimension(ROWS) % size == 0);
        assert(input.dimension(COLS) % size == 0);
        indicators = LongsTensor3D(input.dimension(SLICES), input.dimension(ROWS) / size,
                                   input.dimension(COLS) / size);
        return PoolingLayer::apply(input);
    }
}

#endif //CNN_MAXPOOLINGLAYER_H
