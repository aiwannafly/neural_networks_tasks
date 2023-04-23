#ifndef CNN_SOFTMAXLAYER_H
#define CNN_SOFTMAXLAYER_H

#include "../../eigen.h"

namespace CNN {
    class SoftmaxLayer {

    public:
        Eigen::VectorXf apply(const Eigen::VectorXf& input);
    };
}

#endif //CNN_SOFTMAXLAYER_H
