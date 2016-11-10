//
// File name: Util.h
// Created by ronny on 16-7-15.
// Copyright (c) 2016 SenseNets. All rights reserved.
//

#ifndef NEURALNETWORK_UTIL_H
#define NEURALNETWORK_UTIL_H

#include <string>
#include <opencv2/core/core.hpp>

cv::Mat sigmoid(const cv::Mat &z);
cv::Mat sigmoid_primer(const cv::Mat &z);
void RandomShuffle(cv::Mat &train_data);

enum CostFunction {
    MSE = 0, // mean squared error
    CEE = 1  // cross entropy error
};

cv::Mat cost_derivation(const cv::Mat &a, const cv::Mat &y, CostFunction type);

class Util {
public:
    static bool LoadMNIST(const std::string &prefix,
                          cv::Mat &train_data,
                          cv::Mat &test_data);
};


#endif //NEURALNETWORK_UTIL_H
