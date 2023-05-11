#ifndef RNN_LSTMLAYER_H
#define RNN_LSTMLAYER_H

#include "../Layer.h"
#include "../../common/types.h"

namespace NN {
    class LSTMLayer : public Layer {
    public:
        LSTMLayer(size_t input_size, size_t output_size);

        size_t getInputSize() const override;

        size_t getOutputSize() const override;

        Vector forward(const Vector &input) override;

        Vector backprop(const Vector &input, const Vector& prev_deltas, float l_rate) override;

        ~LSTMLayer() override;

    private:
        size_t input_size;
        size_t output_size;

        Vector curr_hidden;
        Vector curr_cell;
        Params *_forget;
        Params *_input;
        Params *_cell;
        Params *_output;
    };
}

#endif //RNN_LSTMLAYER_H
