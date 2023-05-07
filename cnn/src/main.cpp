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
    for (int i = 1; i < vector.size(); i++) {
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
    while ((s = in.next_line()) != nullptr) {
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
                numVal /= 4;
                example.sample(0, i, j) = numVal;
            }
        }
        examples.push_back(example);
    }
    return examples;
}

struct ClassificationScore {
    int TP = 0;
    int FP = 0;
    int TN = 0;
    int FN = 0;
};

int main(int argc, char *argv[]) {
    if (argc != 1 + 1 + 1) {
        std::cerr << "usage: ./cnn <mnist-train.csv> <mnist-test.csv>" << std::endl;
        return -1;
    }
    int classes_count = 10;
    std::vector<CNN::Example> train_examples = ParseMNISTCSV(argv[1]);
    std::vector<CNN::Example> test_examples = ParseMNISTCSV(argv[2]);
    CNN::LeNet5 nn = CNN::LeNet5(classes_count);

    auto start = std::chrono::steady_clock::now();
    int lim = 5;
    nn.setLearningRate(1 / ((float) lim * (float) train_examples.size()));

    for (int i = 0; i < lim; i++) {
        nn.train(train_examples);
        LOG("Epoch " << i + 1 << " / " << lim << " passed");

        int guessed = 0;
        auto* scores = new ClassificationScore[classes_count];

        for (const auto &example: test_examples) {
            int expected = VectorToLabel(example.expected_output);
            auto p = nn.predict(example.sample);
            int predicted = VectorToLabel(p);
            if (expected == predicted) {
                guessed++;
                scores[expected].TP++;
            } else {
                scores[predicted].FP++;
                scores[expected].FN++;
            }
            for (int c = 0; c < classes_count; c++) {
                if (c == expected || c == predicted) {
                    continue;
                }
                scores[c].TN++;
            }
        }
        for (int c = 0; c < classes_count; c++) {
            LOG("c = " << c);
            LOG("TP = " << scores[c].TP);
            LOG("FP = " << scores[c].FP);
            LOG("TN = " << scores[c].TN);
            LOG("FN = " << scores[c].FN);
        }
        LOG("acc = " << (float) guessed / test_examples.size());
        delete[] scores;
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    LOG("train duration: " << elapsed_seconds.count());

    return 0;
}
