#include "utils.h"

#include <string>
#include <cstring>

namespace utils {
    bool ExtractFloat(const char *buf, float *num) {
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

    float Min(float a, float b) {
        if (a < b) {
            return a;
        }
        return b;
    }

    float Max(float a, float b) {
        if (a > b) {
            return a;
        }
        return b;
    }

    char *ReadStr(FILE *fp, size_t start_capacity) {
        if (fp == nullptr || start_capacity == 0) {
            return nullptr;
        }
        int ch = '\0';
        // skipping all spaces
        while (isspace(ch = fgetc(fp)));
        if (ch == EOF) {
            return nullptr;
        }
        size_t size = start_capacity;
        char *word = (char*) calloc(size, sizeof(*word));
        size_t len = 1;
        while (!isspace(ch) && ch != EOF) {
            word[len - 1] = (char) ch;
            len++;
            if (len >= size) {
                size *= 2;
                char *temp = (char *) realloc(word, size);
                if (temp == nullptr) {
                    free(word);
                    return nullptr;
                } else {
                    word = temp;
                }
            }
            ch = fgetc(fp);
        }
        word[len - 1] = '\0';
        return word;
    }
}
