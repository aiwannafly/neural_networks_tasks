#include "PoolingLayer.h"

namespace cnn {
    PoolingLayer::PoolingLayer(int size) {
        assert(size != 0);
        this->size = size;
    }

    Tensor3D PoolingLayer::apply(const Tensor3D &input) {
        assert(input.dimension(ROWS) % size == 0);
        assert(input.dimension(COLS) % size == 0);
        Tensor3D pooled(input.dimension(SLICES), input.dimension(ROWS) / size,
                        input.dimension(COLS) / size);
        std::array<long, 3> offset = {0, 0, 0}; // Starting point
        std::array<long, 3> extent{};
        extent[SLICES] = 1;
        extent[ROWS] = size;
        extent[COLS] = size;
        for (long slice = 0; slice < input.dimension(SLICES); slice++) {
            offset[SLICES] = slice;
            for (long row = 0; row < input.dimension(ROWS); row += size) {
                offset[ROWS] = row;
                for (long col = 0; col < input.dimension(COLS); col += size) {
                    offset[COLS] = col;
                    float pooledVal = getPool(input.slice(offset, extent));
                    pooled(slice, row / size, col / size) = pooledVal;
                }
            }
        }
        return pooled;
    }
}
