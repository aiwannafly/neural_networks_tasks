#ifndef RNN_LSTMLAYER_H
#define RNN_LSTMLAYER_H

#include "../Layer.h"
#include "../../common/types.h"

namespace NN {
    typedef struct LSTMOutput {
        Vector input;
        Vector i;
        Vector f;
        Vector a;
        Vector o;
        Vector curr_h;
        Vector prev_h;
        Vector curr_c;
        Vector prev_c;
    } LSTMOutput;

    class LSTMLayer {
    public:
        LSTMLayer(size_t input_size, size_t output_size);

        size_t getInputSize() const;

        size_t getOutputSize() const;

        Vector forward(const Vector &input);

        LSTMOutput verboseForward(const Vector &input);

        Vector backprop(const LSTMOutput &o, const Vector& prev_deltas);

        void initBackpropSession();

        void finishBackpropSession(float l_rate);

        ~LSTMLayer();

    private:

        size_t input_size;
        size_t output_size;

        Vector curr_h;
        Vector curr_c;
        Vector curr_delta_h;
        Vector curr_delta_c;
        Vector curr_f;

        LSTMDeltas *_deltas;
        LSTMParams *_forget;
        LSTMParams *_input;
        LSTMParams *_activ;
        LSTMParams *_output;
    };
}

#endif //RNN_LSTMLAYER_H
