#!/bin/bash
clang++ -Wall -pedantic -fsanitize=address main.cpp -lconfigparser utils.cpp MultilayerPerceptron.cpp Layer.cpp -o perceptron

