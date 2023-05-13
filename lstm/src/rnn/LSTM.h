#ifndef RNN_LSTM_H
#define RNN_LSTM_H

#include "../eigen.h"
#include "../common/types.h"
#include "../layers/lstm/LSTMLayer.h"

namespace NN {
    class LSTM {
    public:
        LSTM(size_t input_size, size_t output_size);

        ~LSTM();

        Vector forward(const Vector &input);

        void train(const std::vector<Example> &examples);

        RegressionScore getScore(const std::vector<NN::Example> &examples, float realDispersion);

    private:
        LSTMLayer *lstmLayer;

        size_t input_size;
        size_t output_size;
        float l_rate = 0.1;
    };
}

#endif //RNN_LSTM_H
