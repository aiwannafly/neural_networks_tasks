#ifndef PERCEPTRON_UTILS_H
#define PERCEPTRON_UTILS_H

#include <cstdio>

namespace utils {
    bool ExtractFloat(const char *buf, float *num);

    float Min(float a, float b);

    float Max(float a, float b);

    char *ReadStr(FILE *fp, size_t start_capacity);
}


#endif //PERCEPTRON_UTILS_H
