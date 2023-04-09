#include "MultilayerPerceptron.h"

#include <iostream>
#include <limits>

#include "utils.h"

namespace {

    Eigen::VectorXf mul_inverse(const Eigen::VectorXf &input) {
        Eigen::VectorXf res = input;
        for (int i = 0; i < input.size(); i++) {
            res(i) *= (1 - res(i));
        }
        return res;
    }
}

namespace perceptron {
    MultilayerPerceptron::MultilayerPerceptron(const std::vector<size_t> &layer_sizes, float sigmoid_param) {
        assert(layer_sizes.size() > 1);
        layers = new std::vector<Layer *>();
        this->sigmoid_param = sigmoid_param;
        for (int current = 1; current < layer_sizes.size(); current++) {
            layers->push_back(new Layer(layer_sizes.at(current - 1), layer_sizes.at(current), sigmoid_param));
        }
        layers_deltas = new std::vector<Layer *>();
        for (int current = 1; current < layer_sizes.size(); current++) {
            layers_deltas->push_back(new Layer(layer_sizes.at(current - 1), layer_sizes.at(current), sigmoid_param));
        }
    }

    MultilayerPerceptron::~MultilayerPerceptron() {
        for (auto &layer: *layers) {
            delete layer;
        }
        delete layers;
        for (auto &layer: *layers_deltas) {
            delete layer;
        }
        delete layers_deltas;
    }

    Eigen::VectorXf MultilayerPerceptron::predict(const Eigen::VectorXf &input) {
        Eigen::VectorXf current = input;
        for (auto layer: *layers) {
            current = layer->calculate(current);
        }
        return current;
    }

    float MultilayerPerceptron::get_example_mse(const Eigen::VectorXf &input, const Eigen::VectorXf &expected_output) {
        Eigen::VectorXf output = predict(input);
        auto err_vector = expected_output - output;
        return err_vector.dot(err_vector);
    }

    float MultilayerPerceptron::get_example_mae(const Eigen::VectorXf &input, const Eigen::VectorXf &expected_output) {
        Eigen::VectorXf output = predict(input);
        auto err_vector = expected_output - output;
        float sum = 0;
        for (int i = 0; i < err_vector.size(); i++) {
            sum += abs(err_vector(i));
        }
        return sum;
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

    void MultilayerPerceptron::train(const std::vector<Example> &examples) {
        for (auto layer_delta: *layers_deltas) {
            layer_delta->get_weights()->setZero();
            layer_delta->get_biases()->setZero();
        }
        for (const auto &example: examples) {
            train_example(example.sample, example.target);
        }
        for (int i = 0; i < layers->size(); i++) {
            *(layers->at(i)->get_weights()) += *(layers_deltas->at(i)->get_weights());
            *(layers->at(i)->get_biases()) += *(layers_deltas->at(i)->get_biases());
        }
    }

    void MultilayerPerceptron::train_example(const Eigen::VectorXf &input,
                                             const Eigen::VectorXf &expected_output) { // back_prop
        std::vector<Eigen::VectorXf> *interim_outputs = get_interim_results(input);
        Eigen::VectorXf output = interim_outputs->back();
        Eigen::VectorXf loss_function_deriv = get_loss_function_deriv(expected_output, output);
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
            // sigma'(x) = a*sigma(x)(1 - sigma(x))
            Eigen::VectorXf sigmoid_deriv = sigmoid_param * mul_inverse(current_output);
            delta_curr = Eigen::VectorXf(current_layer_size);
            Eigen::MatrixXf *weights = layers_deltas->at(layer_id)->get_weights();
            Eigen::VectorXf *biases = layers_deltas->at(layer_id)->get_biases();
            if (layer_id == (int) layers->size() - 1) {
                for (int j = 0; j < current_layer_size; j++) {
                    delta_curr(j) = loss_function_deriv(j) * sigmoid_deriv(j);
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
                    delta_curr(j) = scalar_prod * sigmoid_deriv(j);
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

    Eigen::VectorXf MultilayerPerceptron::get_loss_function_deriv(const Eigen::VectorXf &expected_output,
                                                                  const Eigen::VectorXf &predicted_output) {
        if (task == REGRESSION) {
            // - MSE deriv
            return 2 * (expected_output - predicted_output);
        }
        // binary classification, - cross-entropy deriv
        float expected = expected_output(0);
        float probability = predicted_output(0);
        assert(expected == 0 || expected == 1);
        assert(0 <= probability <= 1);
        float loss_deriv = 1;
        if (expected == 0) {
            if (probability == 1) {
                loss_deriv = std::numeric_limits<float>::max();
            } else {
                loss_deriv /= (1 - probability);
            }
        } else {
            if (probability == 0) {
                loss_deriv = std::numeric_limits<float>::min();
            } else {
                loss_deriv /= -probability;
            }
        }
        auto res = Eigen::VectorXf(1);
        res(0) = -loss_deriv;
        return res;
    }

    void MultilayerPerceptron::set_learning_rate(float new_rate) {
        learning_rate = new_rate;
    }

    float MultilayerPerceptron::get_mse(const std::vector<Example> &examples) {
        float sum_err = 0;
        for (const auto &example: examples) {
            sum_err += get_example_mse(example.sample, example.target);
        }
        return sum_err / (float) examples.size();
    }

    float MultilayerPerceptron::get_mae(const std::vector<Example> &examples) {
        float sum_err = 0;
        for (const auto &example: examples) {
            sum_err += get_example_mae(example.sample, example.target);
        }
        return sum_err / (float) examples.size();
    }

    void MultilayerPerceptron::save_weights(FILE *fp) {
        if (fp == nullptr) {
            return;
        }
        for (auto &layer: *layers) {
            fprintf(fp, "%zu %zu\n", layer->get_prev_layer_size(), layer->get_current_layer_size());
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
        while (true) {
            size_t width = 1;
            bool read = utils::ReadSize_t(fp, &width);
            if (!read) {
                break;
            }
            size_t height = 1;
            read = utils::ReadSize_t(fp, &height);
            if (!read) {
                for (auto &l: *new_layers) {
                    delete l;
                }
                delete new_layers;
                return false;
            }
            auto new_layer = new Layer(width, height, sigmoid_param);
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

    void MultilayerPerceptron::set_task_type(Task new_task) {
        this->task = new_task;
    }

    BinaryClassificationScore MultilayerPerceptron::get_classification_scores(const std::vector<Example> &examples) {
        auto res = BinaryClassificationScore();
        for (const auto &example: examples) {
            float predicted = roundf(predict(example.sample)(0));
            float expected = example.target(0);
            if (expected == 0) {
                if (predicted == 0) {
                    res.true_negative++;
                } else {
                    res.false_positive++;
                }
            } else {
                if (predicted == 0) {
                    res.false_negative++;
                } else {
                    res.true_positive++;
                }
            }
        }
        return res;
    }

    float MultilayerPerceptron::get_rscore(const std::vector<Example> &examples) {
        float explained_dispersion = 0;
        for (const auto &example: examples) {
            explained_dispersion += get_example_mse(example.sample, example.target);
        }
        Eigen::VectorXf avg_real = examples.at(0).target;
        avg_real.setZero();
        for (const auto &example: examples) {
            avg_real += example.target;
        }
        avg_real /= (float) examples.size();
        float real_dispersion = 0;
        for (const auto &example: examples) {
            real_dispersion += avg_real.dot(example.target);
        }
        return 1 - explained_dispersion / real_dispersion;
    }
}
