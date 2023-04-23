//
// Created by ms_dr on 20.04.2023.
//

#ifndef CNN_CONVOLUTIONLAYER_H
#define CNN_CONVOLUTIONLAYER_H

#include "../CNNLayer.h"
#include "../../eigen.h"

/*
 * It's a convolution layer. The parameters of a core depend
 * on size of an input tensor, i.e. slices (feature maps) count, so
 * it must be passed through constructor.
 *
 * Core (or a filter) is a tensor of size coreSize * coreSize * inputSlicesCount
 * and one bias. Each core produces one slice (feature map).
 */
namespace cnn {
    class ConvolutionLayer : public CNNLayer {
    public:

        ConvolutionLayer(long coreSize, long coresCount, long inputSlicesCount);

        Tensor3D apply(const Tensor3D &input) override;

        Tensor3D backprop(const Tensor3D &input, const Tensor3D &deltas) override;

        long getCoreSize() const;

        long getCoresCount() const;

        long getInputMapsCount() const;

        Tensor4D *getCores();

        Eigen::VectorXf *getBiases();

        ~ConvolutionLayer() override;
    private:

        long coreSize;
        long inputMapsCount;
        long coresCount;

        // dims of cores:  coresCount * inputMapsCount * coreSize * coreSize
        Tensor4D *cores;
        // dims of biases: coresCount
        Eigen::VectorXf *biases;
    };
}


#endif //CNN_CONVOLUTIONLAYER_H
