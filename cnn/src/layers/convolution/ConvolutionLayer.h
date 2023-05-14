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

        ConvolutionLayer(long f_size, long f_cnt, long input_maps);

        Tensor3D forward(const Tensor3D &input) override;

        Tensor3D backprop(const Tensor3D &input, const Tensor3D &output_deltas, float l_rate) override;

        Matrix weightsTemplate() const override;

        void applyWeightsDeltas(const Matrix &w_deltas) override;

        long getCoreSize() const;

        long getFiltersCount() const;

        long getInputMapsCount() const;

        ~ConvolutionLayer() override;
    private:

        long f_size;
        long input_maps;
        long f_cnt;
        std::array<long, 3> extent{};

        // dims of cores:  coresCount * inputMapsCount * coreSize * coreSize
        Tensor4D *filters;
        // dims of biases: coresCount
        Vector *biases;
    };
}


#endif //CNN_CONVOLUTIONLAYER_H
