#include <iostream>

#include "lenet5/LeNet5.h"
#include "utils/csv.h"

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

ClassificationScore
GetScore(const std::vector<int> &expected_labels, const std::vector<Vector> &predicted_outputs, int label,
         float threshold) {
    assert(expected_labels.size() == predicted_outputs.size());
    auto res = ClassificationScore();
    for (int i = 0; i < expected_labels.size(); i++) {
        float p = predicted_outputs.at(i)(label);
        bool expected = expected_labels.at(i) == label;
        bool predicted = p >= threshold;
        if (expected) {
            if (predicted) {
                res.TP++;
            } else {
                res.FN++;
            }
        } else {
            if (predicted) {
                res.FP++;
            } else {
                res.TN++;
            }
        }
    }
    return res;
}

int main(int argc, char *argv[]) {
    if (argc != 1 + 1 + 1) {
        std::cerr << "usage: ./cnn <mnist-train.csv> <mnist-test.csv>" << std::endl;
        return -1;
    }
    int classes_cnt = 10;
    std::vector<CNN::Example> train_examples = ParseMNISTCSV(argv[1]);
    std::vector<CNN::Example> test_examples = ParseMNISTCSV(argv[2]);
    CNN::LeNet5 nn = CNN::LeNet5(classes_cnt);

    auto start = std::chrono::steady_clock::now();
    int epochs_cnt = 5;
    nn.setLearningRate(1 / ((float) epochs_cnt * (float) train_examples.size()));

    std::vector<int> test_expected_labels;
    std::vector<Vector> test_predicted_outputs;

    for (int i = 0; i < epochs_cnt; i++) {
        nn.train(train_examples);
        LOG("Epoch " << i + 1 << " / " << epochs_cnt << " passed");

        int guessed = 0;
        auto *scores = new ClassificationScore[classes_cnt];

        test_predicted_outputs.clear();
        for (const auto &example: test_examples) {
            int expected = VectorToLabel(example.expected_output);
            if (test_expected_labels.size() < test_examples.size()) {
                test_expected_labels.push_back(expected);
            }
            auto p = nn.predict(example.sample);
            int predicted = VectorToLabel(p);
            test_predicted_outputs.push_back(p);
            if (expected == predicted) {
                guessed++;
                scores[expected].TP++;
            } else {
                scores[predicted].FP++;
                scores[expected].FN++;
            }
            for (int c = 0; c < classes_cnt; c++) {
                if (c == expected || c == predicted) {
                    continue;
                }
                scores[c].TN++;
            }
        }
        for (int c = 0; c < classes_cnt; c++) {
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
    std::chrono::duration<double> elapsed_seconds = end - start;
    LOG("train duration: " << elapsed_seconds.count());

    float step = 0.01;
    // collect data for recall curve
    for (int c = 0; c < classes_cnt; c++) {
        float threshold = 0;
        LOG(c);
        while (threshold <= 1) {
            auto scores = GetScore(test_expected_labels, test_predicted_outputs, c, threshold);
            float fpr = (float) scores.FP / (float) (scores.FP + scores.TN);
            float tpr = (float) scores.TP / (float) (scores.TP + scores.FN);
            std::cout << fpr << " " << tpr << " ";
            threshold += step;
        }
        LOG("");
    }
    return 0;
}
