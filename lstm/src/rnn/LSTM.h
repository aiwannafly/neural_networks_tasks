#ifndef RNN_LSTM_H
#define RNN_LSTM_H

#include "../eigen.h"
#include "../common/types.h"

namespace NN {
    class LSTM {
    public:
        LSTM(int input_size, int output_size);

        Vector forward(const Vector &input);

        void train(const std::vector<Example> &examples);

    private:
        int input_size;
        int output_size;
    };
}

#endif //RNN_LSTM_H
