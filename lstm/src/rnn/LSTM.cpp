#include "LSTM.h"

#include <iostream>

NN::LSTM::LSTM(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;
    lstmLayer = new LSTMLayer(input_size, output_size);
}

Vector NN::LSTM::forward(const Vector &input) {
    return lstmLayer->forward(input);
}

void NN::LSTM::train(const std::vector<Example> &examples) {
    int seqLength = 10;
    for (int e = 0; e < examples.size(); e += seqLength) {
        lstmLayer->initBackpropSession();
        std::vector<LSTMOutput> outputs;
        std::vector<Vector> expected;
        int curr = e;
        while (curr < examples.size() && curr < e + seqLength) {
            outputs.push_back(lstmLayer->verboseForward(examples.at(curr).sample));
            expected.push_back(examples.at(curr++).expected_output);
        }
        curr -= e + 1;
        while (curr >= 0) {
            Vector deltas = outputs.at(curr).curr_h - expected.at(curr);
            lstmLayer->backprop(outputs.at(curr), deltas);
            curr--;
        }
        lstmLayer->finishBackpropSession(l_rate);
    }
}

NN::LSTM::~LSTM() {
    delete lstmLayer;
}
