#include <iostream>

#include "utils/csv.h"
#include "common/functions.h"
#include "utils/utils.h"
#include "common/types.h"
#include "rnn/LSTM.h"

std::vector<std::string> Split(const std::string &s, const std::string &delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

std::vector<float> MakeFloatArr(const std::vector<std::string> &strings) {
    std::vector<float> res;
    for (const auto &str: strings) {
        float num;
        assert(utils::ExtractFloat(str.data(), &num));
        res.push_back(num);
    }
    return res;
}

std::vector<NN::Example> ParseCSV(const std::string &file_name) {
    io::CSVReader<1> in(file_name);
    char *col_names_raw = in.next_line();
    std::string delimiter = ",";
    std::vector<std::string> cols = Split(col_names_raw, delimiter);
    std::vector<NN::Example> examples;
    char *s;
    int sample_offset = 2;
    while ((s = in.next_line()) != nullptr) {
        std::vector<float> nums = MakeFloatArr(Split(s, delimiter));
        if (nums.size() != cols.size()) {
            throw std::invalid_argument("labels count != numbers in row count");
        }
        NN::Example example;
        example.expected_output = Vector(1);
        example.expected_output(0) = nums.at(1);
        int sample_size = (int) nums.size() - sample_offset;
        example.sample = Vector(sample_size);
        for (int i = 0; i < sample_size; i++) {
            example.sample(i) = nums.at(i + sample_offset);
        }
        examples.push_back(example);
    }
    return examples;
}

float GetDataDispersion(const std::vector<NN::Example> &examples) {
    Eigen::VectorXf avg_real = examples.at(0).expected_output;
    avg_real.setZero();
    for (const auto &example: examples) {
        avg_real += example.expected_output;
    }
    avg_real /= (float) examples.size();
    float real_dispersion = 0;
    for (const auto &example: examples) {
        real_dispersion += avg_real.dot(example.expected_output);
    }
    return real_dispersion;
}

int main(int argc, char *argv[]) {
    auto examples = ParseCSV("energy.csv");
    LOG(examples.size());
    auto training_count = (size_t) ((float) examples.size() * .8);
    if (training_count == 0) {
        return -1;
    }
    std::vector<NN::Example> training_examples;
    std::vector<NN::Example> test_examples;
    for (int i = 0; i < examples.size(); i++) {
        if (i < training_count) {
            training_examples.push_back(examples.at(i));
        } else {
            test_examples.push_back(examples.at(i));
        }
    }
    float realDispersion = GetDataDispersion(test_examples);
    FILE *metrics_output = fopen("metrics", "w");
    if (metrics_output == nullptr) {
        LOG_ERR("could not open metrics file");
        return -1;
    }
    fprintf(metrics_output, "MSE, MAE, RSCORE: ");
    auto *nn = new NN::LSTM(examples.front().sample.size(), 1);

    int epoch_cnt = 500;
    for (int i = 0; i < epoch_cnt; i++) {
        nn->train(training_examples);
        auto scores = nn->getScore(test_examples, realDispersion);
        LOG("Epoch " << i + 1 << " passed, " << "MSE: " << scores.MSE);
        fprintf(metrics_output, "%f %f %f ", scores.MSE, scores.MAE, scores.RScore);
    }

    fprintf(metrics_output, "\nPREDICTED, EXPECTED: ");
    for (const auto &example: test_examples) {
        Eigen::VectorXf predicted = nn->forward(example.sample);
        for (int i = 0; i < predicted.size(); i++) {
            fprintf(metrics_output, "%f %f ", predicted(i), example.expected_output(i));
        }
    }
    fclose(metrics_output);
    delete nn;
    return 0;
}
