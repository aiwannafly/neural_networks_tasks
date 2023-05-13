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
    while ((s = in.next_line()) != nullptr) {
        std::vector<float> nums = MakeFloatArr(Split(s, delimiter));
        if (nums.size() != cols.size()) {
            throw std::invalid_argument("labels count != numbers in row count");
        }
        NN::Example example;
        examples.push_back(example);
    }
    return examples;
}

int main(int argc, char *argv[]) {
    auto *nn = new NN::LSTM(2, 1);
    std::vector<NN::Example> examples;
    auto x1 = NN::Example();
    x1.sample = Vector(2);
    x1.expected_output = Vector(1);
    x1.sample(0) = 1;
    x1.sample(1) = 2;
    x1.expected_output(0) = 0.5;
    examples.push_back(x1);
    auto x2 = NN::Example();
    x2.sample = Vector(2);
    x2.expected_output = Vector(1);
    x2.sample(0) = 0.5;
    x2.sample(1) = 3;
    x2.expected_output(0) = 1.25;
    examples.push_back(x2);
    nn->train(examples);
    delete nn;
    return 0;
}
