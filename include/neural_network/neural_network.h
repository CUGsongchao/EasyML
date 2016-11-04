//
// File name: NeuralNetWork.h
// Created by ronny on 16-7-7.
// Copyright (c) 2016 SenseNets. All rights reserved.
//

#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include <vector>
#include <initializer_list>  // std::initializer_list

#include <opencv2/core/core.hpp> // cv::Mat

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<unsigned int> &sizes);
    bool FeedForward(const cv::Mat &input, cv::Mat &output);
    int Predict(const cv::Mat &input);

    bool Train(const cv::Mat &training_samples,
               int epochs,
               int mini_batch_size,
               float eta,
               const cv::Mat &testing_samples = cv::Mat());
private:
    void UpdateMiniBatch(const cv::Mat &subset_data, float eta);
    void BackPropagation(const cv::Mat &x, const cv::Mat &y,
                         std::vector<cv::Mat>& delta_nabla_w,
                         std::vector<cv::Mat>& delta_nabla_b);
    void DataNormalization(cv::Mat &data);
    bool Verify(const cv::Mat &test_case);
private:
    unsigned int num_layers_;
    std::vector<unsigned int> sizes_;
    std::vector<cv::Mat> biases_;
    std::vector<cv::Mat> weights_;
};


#endif //NEURALNETWORK_NEURALNETWORK_H
