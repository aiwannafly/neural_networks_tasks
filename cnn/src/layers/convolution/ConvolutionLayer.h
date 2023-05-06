//
// Created by ms_dr on 20.04.2023.
//

#ifndef CNN_CONVOLUTIONLAYER_H
#define CNN_CONVOLUTIONLAYER_H

#include "../CNNLayer.h"
#include "../../eigen.h"

/*
 * The parameters of a core depend
 * on size of an input tensor, i.e. input slices (feature maps) count, so
 * it must be passed through constructor.
 *
 * Core (or a filter) is a tensor of size coreSize * coreSize * inputSlicesCount
 * and one bias. Each core produces one output slice (feature map).
 */
namespace CNN {
    class ConvolutionLayer : public CNNLayer {
    public:

        ConvolutionLayer(long coreSize, long coresCount, long inputSlicesCount);

        Tensor3D apply(const Tensor3D &input) override;

        Tensor3D backprop(const Tensor3D &input, const Tensor3D &deltas, float l_rate) override;

        long getCoreSize() const;

        long getFiltersCount() const;

        long getInputMapsCount() const;

        Tensor4D *getCores();

//        Eigen::VectorXf *getBiases();

        ~ConvolutionLayer() override;
    private:

        long f_size;
        long inputMapsCount;
        long f_cnt;
        std::array<long, 4> f_extent{};
        std::array<long, 3> extent{};

        // dims of cores:  coresCount * inputMapsCount * coreSize * coreSize
        Tensor4D *filters;
        // dims of biases: coresCount
        Eigen::VectorXf *biases;
    };
}


#endif //CNN_CONVOLUTIONLAYER_H
