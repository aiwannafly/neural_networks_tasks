#include "MultilayerPerceptron.h"

#include <iostream>

#include "utils.h"

namespace {
    float calc_sum_err(const Eigen::VectorXf &diff) {
        return diff.dot(diff);
    }

    Eigen::VectorXf mul_inverse(const Eigen::VectorXf &input) {
        Eigen::VectorXf res = input;
        for (int i = 0; i < input.size(); i++) {
            res(i) *= (1 - res(i));
        }
        return res;
    }
}

namespace perceptron {
    MultilayerPerceptron::MultilayerPerceptron(const std::vector<size_t> &layer_sizes) {
        assert(layer_sizes.size() > 1);
        layers = new std::vector<Layer *>();
        for (int current = 1; current < layer_sizes.size(); current++) {
            layers->push_back(new Layer(layer_sizes.at(current - 1), layer_sizes.at(current)));
        }
    }

    MultilayerPerceptron::~MultilayerPerceptron() {
        for (auto &layer: *layers) {
            delete layer;
        }
        delete layers;
    }

    Eigen::VectorXf MultilayerPerceptron::predict(const Eigen::VectorXf &input) {
        Eigen::VectorXf current = input;
        for (auto layer: *layers) {
            current = layer->calculate(current);
        }
        return current;
    }

    float MultilayerPerceptron::get_example_err(const Eigen::VectorXf &input, const Eigen::VectorXf &expected_output) {
        Eigen::VectorXf output = predict(input);
        auto err_vector = expected_output - output;
        float sum_err = calc_sum_err(err_vector);
        return sum_err;
    }

    std::vector<Eigen::VectorXf> *MultilayerPerceptron::get_interim_results(const Eigen::VectorXf &input) {
        auto results = new std::vector<Eigen::VectorXf>();
        Eigen::VectorXf current = input;
        results->push_back(current);
        for (auto layer: *layers) {
            current = layer->calculate(current);
            results->push_back(current);
        }
        return results;
    }

    void MultilayerPerceptron::train_example(const Eigen::VectorXf &input,
                                             const Eigen::VectorXf &expected_output) { // back_prop
        std::vector<Eigen::VectorXf> *interim_outputs = get_interim_results(input);
        Eigen::VectorXf output = interim_outputs->back();
        auto err_vector = expected_output - output;

        // from the last hidden layer to the input layer
        Eigen::VectorXf delta_prev;
        Eigen::VectorXf delta_curr;

        // some explanation of the code may be found here: https://www.youtube.com/watch?v=tIeHLnjs5U8&t=1s
        for (int layer_id = (int) layers->size() - 1; layer_id >= 0; layer_id--) {
            auto layer = layers->at(layer_id);
            size_t current_layer_size = layer->get_current_layer_size();
            size_t prev_layer_size = layer->get_prev_layer_size();
            Eigen::VectorXf prev_output = interim_outputs->at(layer_id);
            Eigen::VectorXf current_output = interim_outputs->at(layer_id + 1);
            // sigma'(x) = sigma(x)(1 - sigma(x))
            Eigen::VectorXf sigmoid_deriv = mul_inverse(current_output);
            delta_curr = Eigen::VectorXf(current_layer_size);
            Eigen::MatrixXf *weights = layer->get_weights();
            Eigen::VectorXf *biases = layer->get_biases();

            if (layer_id == (int) layers->size() - 1) {
                for (int j = 0; j < current_layer_size; j++) {
                    delta_curr(j) = 2 * err_vector(j) * sigmoid_deriv(j);
                    float common_err = learning_rate * delta_curr(j);
                    for (int k = 0; k < prev_layer_size; k++) {
                        float err = common_err * prev_output(k);
                        (*weights)(j, k) += err;
                    }
                    float err = common_err;
                    (*biases)(j) += err;
                }
            } else {
                auto next_layer = layers->at(layer_id + 1);
                Eigen::MatrixXf *next_weights = next_layer->get_weights();
                for (int j = 0; j < current_layer_size; j++) {
                    Eigen::VectorXf curr_weights = next_weights->col(j);
                    float scalar_prod = curr_weights.dot(delta_prev);
                    delta_curr(j) = 2 * scalar_prod * sigmoid_deriv(j);
                    float common_err = learning_rate * delta_curr(j);
                    for (int k = 0; k < prev_layer_size; k++) {
                        float err = common_err * prev_output(k);
                        (*weights)(j, k) += err;
                    }
                    float err = common_err;
                    (*biases)(j) += err;
                }
            }
            delta_prev = delta_curr;
        }
        delete interim_outputs;
    }

    void MultilayerPerceptron::set_learning_rate(float new_rate) {
        learning_rate = new_rate;
    }

    void MultilayerPerceptron::train(const std::vector<Example> &examples) {
        for (const auto &example: examples) {
            train_example(example.sample, example.target);
        }
    }

    float MultilayerPerceptron::get_err(const std::vector<Example> &examples) {
        float sum_err = 0;
        for (const auto &example: examples) {
            sum_err += get_example_err(example.sample, example.target);
        }
        return sum_err / examples.size();
    }

    void MultilayerPerceptron::save_weights(FILE *fp) {
        if (fp == nullptr) {
            return;
        }
        for (auto &layer: *layers) {
            auto weights = layer->get_weights();
            auto biases = layer->get_biases();
            for (int i = 0; i < weights->rows(); i++) {
                for (int j = 0; j < weights->cols(); j++) {
                    fprintf(fp, "%f ", (*weights)(i, j));
                }
                fprintf(fp, "%f\n", (*biases)(i));
            }
        }
    }

    bool MultilayerPerceptron::read_weights(FILE *fp) {
        if (fp == nullptr) {
            return false;
        }
        auto new_layers = new std::vector<Layer *>();
        for (auto &layer: *layers) {
            size_t width = layer->get_prev_layer_size();
            size_t height = layer->get_current_layer_size();
            auto new_layer = new Layer(width, height);
            auto new_weights = new_layer->get_weights();
            auto new_biases = new_layer->get_biases();
            for (int i = 0; i < height; i++) {
                for (int j = 0; j <= width; j++) {
                    char *str = utils::ReadStr(fp, 10);
                    if (str == nullptr) {
                        cancel:
                        delete new_layer;
                        for (auto &l: *new_layers) {
                            delete l;
                        }
                        delete new_layers;
                        return false;
                    }
                    float num = 0.0;
                    bool extracted = utils::ExtractFloat(str, &num);
                    free(str);
                    if (!extracted) {
                        goto cancel;
                    }
                    if (j == width) {
                        (*new_biases)(i) = num;
                    } else {
                        (*new_weights)(i, j) = num;
                    }
                }
            }
            new_layers->push_back(new_layer);
        }
        for (auto &layer: *layers) {
            delete layer;
        }
        delete layers;
        layers = new_layers;
        return true;
    }
}
