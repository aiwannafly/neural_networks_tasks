#include <iostream>
#include "NeuralNetwork.h"

namespace {
    float calc_sum_err(Eigen::VectorXf diff) {
        float sum = 0;
        for (int i = 0; i < diff.size(); i++) {
            sum += diff(i) * diff(i);
        }
        return sum;
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
    NeuralNetwork::NeuralNetwork(const std::vector<size_t>& layer_sizes) {
        assert(layer_sizes.size() > 1);
        layers = new std::vector<Layer*>();
        for (int current = 1; current < layer_sizes.size(); current++){
            layers->push_back(new Layer(layer_sizes.at(current - 1), layer_sizes.at(current)));
        }
    }

    NeuralNetwork::~NeuralNetwork() {
        for (auto & layer : *layers) {
            delete layer;
        }
        delete layers;
    }

    Eigen::VectorXf NeuralNetwork::calculate(const Eigen::VectorXf &input) {
        Eigen::VectorXf current = input;
        for (auto layer: *layers) {
            current = layer->calculate(current);
        }
        return current;
    }

    float NeuralNetwork::calculate_err(const Eigen::VectorXf &input, const Eigen::VectorXf &expected_output) {
        Eigen::VectorXf output = calculate(input);
        auto err_vector = expected_output - output;
        float sum_err = calc_sum_err(err_vector);
        return sum_err;
    }

    std::vector<Eigen::VectorXf> *NeuralNetwork::calculate_and_get_interim_results(const Eigen::VectorXf &input) {
        auto results = new std::vector<Eigen::VectorXf>();
        Eigen::VectorXf current = input;
        results->push_back(current);
        for (auto layer: *layers) {
            current = layer->calculate(current);
            results->push_back(current);
        }
        return results;
    }

    void NeuralNetwork::train(const Eigen::VectorXf &input, const Eigen::VectorXf &expected_output) {
        std::vector<Eigen::VectorXf> *interim_outputs = calculate_and_get_interim_results(input);
        Eigen::VectorXf output = interim_outputs->back();
        auto err_vector = expected_output - output;
        // float sum_err = calc_sum_err(err_vector);
        // std::cout << "Err: " << sum_err << std::endl;
        // from the last hidden layer to the input layer
        Eigen::VectorXf delta_prev;
        Eigen::VectorXf delta_curr;
        for (int layer_id = (int) layers->size() - 1; layer_id >= 0; layer_id--) {
            auto layer = layers->at(layer_id);
            size_t current_layer_size = layer->get_current_layer_size();
            size_t prev_layer_size = layer->get_prev_layer_size();
            Eigen::VectorXf prev_output = interim_outputs->at(layer_id);
            Eigen::VectorXf current_output = interim_outputs->at(layer_id + 1);
            // sigma'(x) = sigma(x)(1 - sigma(x))
            Eigen::VectorXf sigmoid_deriv = mul_inverse(current_output);
            delta_curr = Eigen::VectorXf(current_layer_size);
//            std::cout << "layer: " << layer_id << std::endl;
            Eigen::MatrixXf *weights = layer->get_weights();
            Eigen::VectorXf *biases = layer->get_biases();
            if (layer_id == (int) layers->size() - 1) {
                for (int j = 0; j < current_layer_size; j++) {
                    delta_curr(j) = err_vector(j) * sigmoid_deriv(j);
                    float common_err = 2 * learning_rate * delta_curr(j);
                    for (int k = 0; k < prev_layer_size; k++) {
                        float err = common_err * prev_output(k);
                        (*weights)(j, k) += err;
//                        std::cout << j << " " << k << " " << err << std::endl;
                    }
                    float err = common_err;
                    (*biases)(j) += err;
//                    std::cout << j << " " << err << std::endl;
                }
            } else {
                auto next_layer = layers->at(layer_id + 1);
                Eigen::MatrixXf *next_weights = next_layer->get_weights();
                for (int j = 0; j < current_layer_size; j++) {
                    Eigen::VectorXf curr_weights = next_weights->col(j);
                    float scalar_prod = curr_weights.dot(delta_prev);
                    delta_curr(j) = scalar_prod * sigmoid_deriv(j);
                    float common_err = 2 * learning_rate * delta_curr(j);
                    for (int k = 0; k < prev_layer_size; k++) {
                        float err = common_err * prev_output(k);
                        (*weights)(j, k) += err;
//                        std::cout << j << " " << k << " " << err << std::endl;
                    }
                    float err = common_err;
                    (*biases)(j) += err;
//                    std::cout << j << " " << err << std::endl;
                }
            }
            delta_prev = delta_curr;
        }
        delete interim_outputs;
    }

    void NeuralNetwork::set_learning_rate(float new_rate) {
        learning_rate = new_rate;
    }
}
