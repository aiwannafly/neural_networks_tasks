#include "PoolingLayer.h"

namespace CNN {
    PoolingLayer::PoolingLayer(int size) {
        assert(size != 0);
        this->size = size;
        extent[MAPS] = 1;
        extent[ROWS] = size;
        extent[COLS] = size;
    }

    Tensor3D PoolingLayer::forward(const Tensor3D &input) {
        assert(input.dimension(ROWS) % size == 0);
        assert(input.dimension(COLS) % size == 0);
        Tensor3D pooled(input.dimension(MAPS), input.dimension(ROWS) / size,
                        input.dimension(COLS) / size);
        std::array<long, 3> offset = {0, 0, 0};
        for (long slice = 0; slice < input.dimension(MAPS); slice++) {
            offset[MAPS] = slice;
            for (long row = 0; row < input.dimension(ROWS); row += size) {
                offset[ROWS] = row;
                for (long col = 0; col < input.dimension(COLS); col += size) {
                    offset[COLS] = col;
                    float pooledVal = getPool(input, offset);
                    pooled(slice, row / size, col / size) = pooledVal;
                }
            }
        }
        return pooled;
    }
}
