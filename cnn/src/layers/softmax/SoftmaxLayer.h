#ifndef CNN_SOFTMAXLAYER_H
#define CNN_SOFTMAXLAYER_H

#include "../../eigen.h"

namespace CNN {
    class SoftmaxLayer {

    public:
        static Vector forward(const Vector& input);

        static Vector deriv(const Vector& output);
    };
}

#endif //CNN_SOFTMAXLAYER_H
