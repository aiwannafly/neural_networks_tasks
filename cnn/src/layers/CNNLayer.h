#ifndef CNN_CNNLAYER_H
#define CNN_CNNLAYER_H

#include "../eigen.h"

namespace cnn {
    class CNNLayer {
    public:
        virtual Tensor3D apply(const Tensor3D &input) = 0;

        virtual ~CNNLayer() = default;
    };
}

#endif //CNN_CNNLAYER_H
