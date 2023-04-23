#ifndef PERCEPTRON_UTILS_H
#define PERCEPTRON_UTILS_H

#include <cstdio>

#define MAX(a, b) (a < b) ? b : a

namespace utils {
    bool ExtractSize_t(const char *buf, size_t *num);

    bool ExtractFloat(const char *buf, float *num);

    float Minf(float a, float b);

    float Maxf(float a, float b);

    long MinL(long a, long b);

    long MaxL(long a, long b);

    bool ReadFloat(FILE *fp, float *num);

    bool ReadSize_t(FILE *fp, size_t *num);

    char *ReadStr(FILE *fp, size_t start_capacity);
}


#endif //PERCEPTRON_UTILS_H
