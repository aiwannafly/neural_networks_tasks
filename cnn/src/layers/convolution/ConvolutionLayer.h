//
// Created by ms_dr on 20.04.2023.
//

#ifndef CNN_CONVOLUTIONLAYER_H
#define CNN_CONVOLUTIONLAYER_H

#include "../CNNLayer.h"
#include "../../eigen.h"

namespace cnn {
    class ConvolutionLayer : public CNNLayer {
    public:

        ConvolutionLayer(size_t coreSize, size_t outputMapsCount, size_t inputMapsCount);

        std::vector<FeatureMap*> *apply(std::vector<FeatureMap*> *maps) override;

        size_t getCoreSize() const;

        size_t getOutputMapsCount() const;

        size_t getInputMapsCount() const;

        std::vector<Eigen::MatrixXf*> *getCores();

        ~ConvolutionLayer() override;
    private:
        FeatureMap *applyCore(FeatureMap *map, Eigen::MatrixXf *core) const;

        size_t coreSize;
        size_t inputMapsCount;

        std::vector<Eigen::MatrixXf*> *cores;
    };
}


#endif //CNN_CONVOLUTIONLAYER_H
