#ifndef CNN_CNNLAYER_H
#define CNN_CNNLAYER_H

#include "../eigen.h"
#include "../feature_map/FeatureMap.h"

namespace cnn {
    class CNNLayer {
    public:
        virtual std::vector<FeatureMap*> *apply(std::vector<FeatureMap*> *maps) = 0;

        virtual ~CNNLayer() = default;
    };
}

#endif //CNN_CNNLAYER_H
