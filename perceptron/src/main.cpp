#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <configparser.hpp>

#include "NeuralNetworkImpl.h"
#include "csv.h"

#define FAIL (-1)
#define SUCCESS (0)

namespace {
    bool extract_float(const char *buf, float *num) {
        if (nullptr == buf || num == nullptr) {
            return false;
        }
        char *end_ptr = nullptr;
        *num = strtof(buf, &end_ptr);
        if (buf + strlen(buf) > end_ptr) {
            return false;
        }
        return true;
    }

    float min(float a, float b) {
        if (a < b) {
            return a;
        }
        return b;
    }

    float max(float a, float b) {
        if (a > b) {
            return a;
        }
        return b;
    }

    std::vector<std::string> split(const std::string &s, const std::string &delimiter) {
        size_t pos_start = 0, pos_end, delim_len = delimiter.length();
        std::string token;
        std::vector<std::string> res;

        while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
            token = s.substr(pos_start, pos_end - pos_start);
            pos_start = pos_end + delim_len;
            res.push_back (token);
        }

        res.push_back (s.substr (pos_start));
        return res;
    }

    std::vector<float> to_float_arr(const std::vector<std::string> &strings) {
        std::vector<float> res;
        for (const auto &str: strings) {
            float num;
            assert(extract_float(str.data(), &num));
            res.push_back(num);
        }
        return res;
    }

    typedef struct Params {
        float learning_rate{};
        float training_percentage{};
        size_t epochs_count{};
        std::string csv_file_name;
        size_t log_frequency{};
        std::string errs_file_name;
        std::vector<size_t> inner_layer_sizes;
        std::vector<std::string> target_names;
        bool ignore_first_col{};
    } Params;

    Params *get_params(const std::string &config_path) {
        auto p = new Params();
        const std::string section_name = "Section1";
        ConfigParser parser = ConfigParser(config_path);
        p->learning_rate = parser.aConfig<float>(section_name, "learning_rate");
        p->training_percentage = parser.aConfig<float>(section_name, "training_percentage");
        p->epochs_count = parser.aConfig<size_t>(section_name, "epochs_count");
        p->csv_file_name = parser.aConfig<std::string>(section_name, "csv_file_name");
        p->log_frequency = parser.aConfig<size_t>(section_name, "log_frequency");
        p->errs_file_name = parser.aConfig<std::string>(section_name, "log_file_name");
        p->inner_layer_sizes = parser.aConfigVec<size_t>(section_name, "inner_layer_sizes");
        p->target_names = parser.aConfigVec<std::string>(section_name, "targets");
        p->ignore_first_col = parser.aConfig<bool>(section_name, "ignore_first_col");
        return p;
    }

    std::vector<perceptron::Example> extract_data_vectors(Params *p) {
        io::CSVReader<1> in(p->csv_file_name);
        char *col_names_raw = in.next_line();
        std::string delimiter = ",";
        std::vector<std::string> cols = split(col_names_raw, delimiter);
        std::vector<size_t> target_ids(p->target_names.size());
        size_t target_len = 0;
        for (size_t i = 0; i < cols.size(); i++) {
            for (size_t j = 0; j < p->target_names.size(); j++) {
                if (p->target_names[j] == cols[i]) {
                    target_ids[j] = i;
                    target_len++;
                }
            }
        }
        if (target_len != p->target_names.size()) {
            throw std::invalid_argument("could not find all targets in .csv file");
        }
        size_t cols_count = cols.size();
        if (cols_count <= target_len + p->ignore_first_col) {
            throw std::invalid_argument("size of a sample is less or equal to zero");
        }
        size_t sample_len = cols_count - target_len - p->ignore_first_col;
        std::vector<perceptron::Example> examples;
        std::vector<float> min_targets(target_len);
        std::vector<float> max_targets(target_len);
        char *s;
        while((s = in.next_line()) != nullptr) {
            std::vector<float> nums = to_float_arr(split(s, delimiter));
            perceptron::Example example;
            example.sample = Eigen::VectorXf(sample_len);
            example.target = Eigen::VectorXf(target_len);
            for (int i = 0; i < target_len; i++) {
                float target = nums.at(target_ids.at(i));
                example.target(i) = target;
                if (examples.empty()) {
                    min_targets[i] = target;
                    max_targets[i] = target;
                } else {
                    min_targets[i] = min(target, min_targets[i]);
                    max_targets[i] = max(target, max_targets[i]);
                }
            }
            int current = 0;
            for (int i = p->ignore_first_col; i < cols_count; i++) {
                bool is_target_idx = false;
                for (int j = 0; j < target_len; j++) {
                    if (i == target_ids.at(j)) {
                        is_target_idx = true;
                        break;
                    }
                }
                if (!is_target_idx) {
                    example.sample(current++) = nums.at(i);
                }
            }
            examples.push_back(example);
        }
        for (int j = 0; j < examples.size(); j++) {
            for (int i = 0; i < target_len; i++) {
                examples[j].target(i) = (examples[j].target(i) - min_targets[i]) / (max_targets[i] - min_targets[i]);
            }
        }
        return examples;
    }
}

int main(int argc, char *argv[]) {
    // RMSE Rscore
    const std::string usage_guide = "usage: ./prog <config.ini>";
    if (argc != 2) {
        std::cerr << usage_guide << std::endl;
        return FAIL;
    }
    const std::string config_path = argv[1];
    Params *p = get_params(config_path);
    std::vector<perceptron::Example> examples = extract_data_vectors(p);
    std::cout << "samples count: " << examples.size() << std::endl;
    auto training_count = (size_t) (examples.size() * p->training_percentage);
    if (training_count == 0) {
        std::cerr << "too little samples" << std::endl;
        delete p;
        return FAIL;
    }
    std::vector<perceptron::Example> training_examples;
    std::vector<perceptron::Example> test_examples;
    for (int i = 0; i < examples.size(); i++) {
        if (i < training_count) {
            training_examples.push_back(examples.at(i));
        } else {
            test_examples.push_back(examples.at(i));
        }
    }
    size_t input_size = examples.begin().base()->sample.size();
    size_t output_size = examples.begin().base()->target.size();
    std::vector<size_t> layer_sizes;
    layer_sizes.push_back(input_size);
    for (auto inner_size: p->inner_layer_sizes) {
        std::cout << inner_size << std::endl;
        layer_sizes.push_back(inner_size);
    }
    layer_sizes.push_back(output_size);
    perceptron::NeuralNetwork *nn = new perceptron::NeuralNetworkImpl(layer_sizes);
    nn->set_learning_rate(p->learning_rate);
    FILE *output = fopen(p->errs_file_name.data(), "w");
    // precision, recall, roc auc, f1, accuracy
    // explain these metrics
    // why we need activation function
    for (int e = 0; e < p->epochs_count; e++) {
        nn->train(training_examples);
        float sum_err = nn->get_err(test_examples);
        if (e % p->log_frequency == 0 || e == p->epochs_count - 1) {
            fprintf(output, "%f ", sum_err);
            std::cout << "Epoch " << e << " : err = " << sum_err << std::endl;
        }
    }
    fclose(output);
    delete p;
    delete nn;
    return SUCCESS;
}
