#!/bin/bash
clang++ -Wall -pedantic -fsanitize=address main.cpp NeuralNetwork.cpp Layer.cpp -o perceptron

