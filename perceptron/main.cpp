#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include "NeuralNetwork.h"
#include "csv.h"

namespace {
    bool extract_int(const char *buf, int *num) {
        if (nullptr == buf || num == nullptr) {
            return false;
        }
        char *end_ptr = nullptr;
        *num = (int) strtol(buf, &end_ptr, 10);
        if (buf + strlen(buf) > end_ptr) {
            return false;
        }
        return true;
    }

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

    std::vector<std::string> split(std::string s, std::string delimiter) {
        size_t pos_start = 0, pos_end, delim_len = delimiter.length();
        std::string token;
        std::vector<std::string> res;

        while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
            token = s.substr (pos_start, pos_end - pos_start);
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
}

int main(int argc, char *argv[]) {
    const std::string usage_guide = "usage: ./prog <data.csv> <epochs_count> <optional: learning_rate>";
    if (argc < 1 + 2 || argc > 1 + 3) {
        std::cerr << usage_guide << std::endl;
        return -1;
    }
    int epochs_count;
    bool extracted = extract_int(argv[2], &epochs_count);
    if (!extracted) {
        std::cerr << "error while parsing epochs_count" << std::endl;
        std::cerr << usage_guide << std::endl;
        return -1;
    }
    float learning_rate = 0.001;
    if (argc > 1 + 2) {
        extracted = extract_float(argv[3], &learning_rate);
        if (!extracted) {
            std::cerr << "error while parsing learning_rate" << std::endl;
            std::cerr << usage_guide << std::endl;
            return -1;
        }
    }
    io::CSVReader<1> in(argv[1]);
    char *col_names_raw = in.next_line();
    std::string delimiter = ",";
    std::vector<std::string> cols = split(col_names_raw, delimiter);    
    int cols_count = cols.size();
    char *s;
    std::vector<Eigen::VectorXf> samples;
    std::vector<float> targets;
    float min_target = 10e10;
    float max_target = -10e10;
    while((s = in.next_line()) != nullptr){
        std::vector<float> nums = to_float_arr(split(s, delimiter));
        Eigen::VectorXf sample = Eigen::VectorXf(cols_count - 2);
        for (int i = 1; i < cols_count - 1; i++) {
            sample(i - 1) = nums.at(i);
        }
        samples.push_back(sample);
        float target = nums.at(cols_count - 1);
        targets.push_back(target);
        if (target > max_target) {
            max_target = target;
        }
        if (target < min_target) {
            min_target = target;
        }
    }
    std::cout << "samples count: " << samples.size() << std::endl;
    size_t training_count = (size_t) (samples.size() * 0.7);
    if (training_count == 0) {
        std::cerr << "too little samples" << std::endl;
        return -2;
    }
    size_t layers_count = 1 + 2 + 1;
    std::vector<size_t> layer_sizes;
    for (size_t i = 0; i < layers_count; i++) {
        if (i == 0 || i == 1) {
            layer_sizes.push_back(cols_count - 2);
        } else if (i == layers_count - 1) {
            layer_sizes.push_back(1);
        } else {
            size_t prev = layer_sizes.at(i - 1);
            size_t new_size = prev / 2;
            if (new_size < 2) {
                new_size = 2;
            }
            layer_sizes.push_back(new_size);
        }
    }
    auto nn = new perceptron::NeuralNetwork(layer_sizes);
    nn->set_learning_rate(learning_rate);
    int log_frequency = 10;
    FILE *output = fopen("err_data", "w");
    for (int e = 0; e < epochs_count; e++) {
        for (int i = 0; i < training_count; i++) {
            float target = targets.at(i);
            target = (target - min_target) / (max_target - min_target); // normalize
            Eigen::VectorXf expected(1);
            expected(0) = target;
            nn->train(samples.at(i), expected);
        }
        float sum_err = 0.0;
        for (int i = training_count; i < samples.size(); i++) {
            float target = targets.at(i);
            target = (target - min_target) / (max_target - min_target); // normalize
            Eigen::VectorXf expected(1);
            expected(0) = target;
            float err = nn->calculate_err(samples.at(i), expected);
            sum_err += err;
        }
        if (e % log_frequency == 0) {
            fprintf(output, "%f ", sum_err);
        }
        std::cout << "Epoch " << e + 1 << " : err = " << sum_err << std::endl;
    }
    fprintf(output, "\n");
    fclose(output);
    delete nn;
    return 0;
}
