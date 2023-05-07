#include <iostream>

#include "lenet5/LeNet5.h"
#include "utils/csv.h"
#include "common/functions.h"

#define IMG_SIZE (28)

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

Vector LabelToVector(int label) {
    Vector result(10);
    result.setZero();
    result(label) = 1;
    return result;
}

int VectorToLabel(Vector vector) {
    float max = vector(0);
    int id = 0;
    for (int i = 1; i < 10; i++) {
        if (vector(i) > max) {
            max = vector(i);
            id = i;
        }
    }
    return id;
}

std::vector<CNN::Example> ParseMNISTCSV(const std::string &file_name) {
    io::CSVReader<1> in(file_name);
    char *col_names_raw = in.next_line();
    std::string delimiter = ",";
    std::vector<std::string> cols = Split(col_names_raw, delimiter);
    std::vector<CNN::Example> examples;
    char *s;
    int count = 0;
    while ((s = in.next_line()) != nullptr) {
        count++;
        if (count > 10) {
            break;
        }
        std::vector<float> nums = MakeFloatArr(Split(s, delimiter));
        if (nums.size() != cols.size()) {
            throw std::invalid_argument("labels count != numbers in row count");
        }
        float label = nums.at(0);
        CNN::Example example;
        example.expected_output = LabelToVector((int) label);
        example.sample = Tensor3D(1, IMG_SIZE, IMG_SIZE);
        for (int i = 0; i < IMG_SIZE; i++) {
            for (int j = 0; j < IMG_SIZE; j++) {
                float numVal = nums.at(1 + j + i * IMG_SIZE) / 255;
                numVal -= 0.5;
                numVal /= 2;
                example.sample(0, i, j) = numVal;
            }
        }
        examples.push_back(example);
    }
    return examples;
}

int main(int argc, char *argv[]) {
    if (argc != 1 + 1 + 1) {
        std::cerr << "usage: ./cnn <mnist-train.csv> <mnist-test.csv>" << std::endl;
        return -1;
    }
    std::vector<CNN::Example> all_examples = ParseMNISTCSV(argv[1]);
    CNN::LeNet5 nn = CNN::LeNet5(10);

//    auto example = all_examples.at(0);
//    PrintVector(example.expected_output);
//    for (int i = 0; i < 100; i++) {
//        // 14 - loss: 2.72768 with no cnn backprop
//        // 14 - loss: 2.48222 with just 1 layer cnn backprop (w -= delta)
//        // 14 - loss: 2.34262 with all cnn layers backprop (w -= delta)
//        // 14 - loss: 2.34262 with all cnn layers backprop (w += delta and inverse first delta)
//        nn.trainExample(example);
//        auto predicted = nn.predict(example.sample);
//        std::cout << i << " - loss: " << CrossEntropy(example.expected_output, predicted) << std::endl;
//    }
//    PrintVector(nn.predict(example.sample));
//    PrintVector(example.expected_output);

    int lim = 10;
    for (int i = 0; i < lim; i++) {
        nn.train(all_examples);
        std::cout << i << "/" << lim << std::endl;
    }
    for (const auto &example: all_examples) {
        PrintVector(example.expected_output);
        PrintVector(nn.predict(example.sample));
        std::cout << std::endl;
    }
    return 0;
}
