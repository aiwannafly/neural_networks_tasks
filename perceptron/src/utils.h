#ifndef PERCEPTRON_UTILS_H
#define PERCEPTRON_UTILS_H

#include <cstdio>

namespace utils {
    bool ExtractSize_t(const char *buf, size_t *num);

    bool ExtractFloat(const char *buf, float *num);

    float Min(float a, float b);

    float Max(float a, float b);

    bool ReadFloat(FILE *fp, float *num);

    bool ReadSize_t(FILE *fp, size_t *num);

    char *ReadStr(FILE *fp, size_t start_capacity);
}


#endif //PERCEPTRON_UTILS_H
