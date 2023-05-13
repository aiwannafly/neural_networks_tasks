#include "LSTMLayer.h"

#include <iostream>

#include "../../common/functions.h"

NN::LSTMLayer::LSTMLayer(size_t input_size, size_t output_size) {
    this->input_size = input_size;
    this->output_size = output_size;
    _forget = new LSTMParams(input_size, output_size);
    _input = new LSTMParams(input_size, output_size);
    _activ = new LSTMParams(input_size, output_size);
    _output = new LSTMParams(input_size, output_size);
    _deltas = new LSTMDeltas(input_size, output_size);
    curr_h = Vector(output_size);
    curr_h.setZero();
    curr_c = Vector(output_size);
    curr_c.setZero();
    curr_delta_h = Vector(output_size);
    curr_delta_h.setZero();
    curr_delta_c = Vector(output_size);
    curr_delta_c.setZero();
    curr_f = Vector(output_size);
    curr_f.setZero();

    _activ->hx(0, 0) = 0.45;
    _activ->hx(0, 1) = 0.25;
    _activ->hh(0, 0) = 0.15;
    _activ->biases(0) = 0.2;

    _input->hx(0, 0) = 0.95;
    _input->hx(0, 1) = 0.8;
    _input->hh(0, 0) = 0.8;
    _input->biases(0) = 0.65;

    _forget->hx(0, 0) = 0.7;
    _forget->hx(0, 1) = 0.45;
    _forget->hh(0, 0) = 0.1;
    _forget->biases(0) = 0.15;

    _output->hx(0, 0) = 0.6;
    _output->hx(0, 1) = 0.4;
    _output->hh(0, 0) = 0.25;
    _output->biases(0) = 0.1;
}

NN::LSTMLayer::~LSTMLayer() {
    delete _forget;
    delete _input;
    delete _activ;
    delete _output;
    delete _deltas;
}

Vector NN::LSTMLayer::forward(const Vector &input) {
    return verboseForward(input).curr_h;
}

size_t NN::LSTMLayer::getInputSize() const {
    return input_size;
}

size_t NN::LSTMLayer::getOutputSize() const {
    return output_size;
}

NN::LSTMOutput NN::LSTMLayer::verboseForward(const Vector &input) {
    auto res = LSTMOutput();
    res.input = input;
    res.f = SigmoidV(_forget->hx * input + _forget->hh * curr_h + _forget->biases);
    res.i = SigmoidV(_input->hx * input + _input->hh * curr_h + _input->biases);
    res.a = TanhV(_activ->hx * input + _activ->hh * curr_h + _activ->biases);
    res.o = SigmoidV(_output->hx * input + _output->hh * curr_h + _output->biases);
    res.prev_c = curr_c;
    res.prev_h = curr_h;
    curr_c = res.f.cwiseProduct(curr_c) + res.i.cwiseProduct(res.a);
    curr_h = res.o.cwiseProduct(TanhV(curr_c));
    res.curr_h = curr_h;
    res.curr_c = curr_c;
    return res;
}

Vector NN::LSTMLayer::backprop(const NN::LSTMOutput &o, const Vector &prev_deltas) {
    size_t n = o.curr_h.size();
    PrintVector(prev_deltas);
    Vector full_delta = prev_deltas + curr_delta_h;
    curr_delta_c = full_delta.cwiseProduct(o.o).cwiseProduct(Ones(n) - TanhV(o.curr_c).cwisePow(2)) +
                   curr_delta_c.cwiseProduct(curr_f);
    curr_f = o.f;
    Vector delta_a = curr_delta_c.cwiseProduct(o.i).cwiseProduct(Ones(o.a.size()) - o.a.cwisePow(2));
    Vector delta_i = curr_delta_c.cwiseProduct(o.a).cwiseProduct(o.i).cwiseProduct(
            Ones(n) - o.i);
    Vector delta_f = curr_delta_c.cwiseProduct(o.prev_c).cwiseProduct(o.f).cwiseProduct(
            Ones(n) - o.f);
    Vector delta_o = full_delta.cwiseProduct(TanhV(o.curr_c)).cwiseProduct(o.o).cwiseProduct(
            Ones(n) - o.o);
    Matrix delta_gates = Matrix(4, n);
    delta_gates.row(0) = delta_a;
    delta_gates.row(1) = delta_i;
    delta_gates.row(2) = delta_f;
    delta_gates.row(3) = delta_o;
    Matrix W = Matrix(n * 4, o.input.size());
    W << _activ->hx, _input->hx, _forget->hx, _output->hx;
    Matrix U = Matrix(n * 4, n);
    U << _activ->hh, _input->hh, _forget->hh, _output->hh;
    Vector delta_input = W.transpose() * delta_gates;
    curr_delta_h = U.transpose() * delta_gates;
    _deltas->W += delta_gates * o.input.transpose();
    _deltas->U += delta_gates * o.prev_h.transpose();
    _deltas->biases += delta_gates;
    return delta_input;
}

void NN::LSTMLayer::initBackpropSession() {
    _deltas->clear();
    curr_delta_h.setZero();
    curr_delta_c.setZero();
    curr_f.setZero();
}

void NN::LSTMLayer::finishBackpropSession(float l_rate) {
    Matrix W = Matrix(output_size * 4, input_size);
    W << _activ->hx, _input->hx, _forget->hx, _output->hx;
    Matrix U = Matrix(output_size * 4, output_size);
    U << _activ->hh, _input->hh, _forget->hh, _output->hh;
    Matrix biases = Matrix(4, output_size);
    biases.row(0) = _activ->biases;
    biases.row(1) = _input->biases;
    biases.row(2) = _forget->biases;
    biases.row(3) = _output->biases;
    W -= _deltas->W * l_rate;
    U -= _deltas->U * l_rate;
    biases -= _deltas->biases * l_rate;
    for (int j = 0; j < output_size; j++) {
        _activ->hx.row(j) = W.row(j);
        _activ->hh.row(j) = U.row(j);
    }
    _activ->biases = biases.row(0);
    for (int j = 0; j < output_size; j++) {
        _input->hx.row(j) = W.row(j + (int) output_size);
        _input->hh.row(j) = U.row(j + (int) output_size);
    }
    _input->biases = biases.row(1);
    for (int j = 0; j < output_size; j++) {
        _forget->hx.row(j) = W.row(j + 2 * (int) output_size);
        _forget->hh.row(j) = U.row(j + 2 * (int) output_size);
    }
    _forget->biases = biases.row(2);
    for (int j = 0; j < output_size; j++) {
        _output->hx.row(j) = W.row(j + 3 * (int) output_size);
        _output->hh.row(j) = U.row(j + 3 * (int) output_size);
    }
    _output->biases = biases.row(3);
}
