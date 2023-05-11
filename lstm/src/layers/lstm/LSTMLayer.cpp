#include "LSTMLayer.h"

#include "../../common/functions.h"

NN::LSTMLayer::LSTMLayer(size_t input_size, size_t output_size) {
    this->input_size = input_size;
    this->output_size = output_size;
    size_t total_size = input_size + output_size;
    _forget = new Params(total_size, output_size);
    _input = new Params(total_size, output_size);
    _cell = new Params(total_size, output_size);
    _output = new Params(total_size, output_size);
    curr_hidden = Vector(output_size);
    curr_hidden.setZero();
    curr_cell = Vector(output_size);
    curr_cell.setZero();
}

NN::LSTMLayer::~LSTMLayer() {
    delete _forget;
    delete _input;
    delete _cell;
    delete _output;
}

Vector NN::LSTMLayer::forward(const Vector &input) {
    auto f = SigmoidV(_forget->weights * input + _forget->biases);
    auto i = SigmoidV(this->_input->weights * input + this->_input->biases);
    auto c = TanhV(_cell->weights * input + _cell->biases);
    auto o = SigmoidV(_output->weights * input + _output->biases);
    curr_cell = f.cwiseProduct(curr_cell) + i.cwiseProduct(c);
    curr_hidden = o.cwiseProduct(TanhV(curr_cell));
    return curr_hidden;
}

Vector NN::LSTMLayer::backprop(const Vector &input, const Vector &prev_deltas, float l_rate) {
    return Vector(input_size);
}

size_t NN::LSTMLayer::getInputSize() const {
    return input_size;
}

size_t NN::LSTMLayer::getOutputSize() const {
    return output_size;
}
