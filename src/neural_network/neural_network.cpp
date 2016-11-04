//
// File name: NeuralNetwork.cpp
// Created by ronny on 16-7-7.
// Copyright (c) 2016 SenseNets. All rights reserved.
//

#include <iostream>
#include <fstream>

#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>

#include "neural_network/neural_network.h"
#include "util/util.h"

NeuralNetwork::NeuralNetwork(const std::vector<unsigned int> &sizes) {
    sizes_ = sizes;
    num_layers_ = sizes_.size();
    CHECK(num_layers_ >= 2)
        << "The number of network layers must be at least 2." << std::endl;
    biases_.assign(num_layers_ - 1, cv::Mat());
    weights_.assign(num_layers_ - 1, cv::Mat());
    for (unsigned i = 1; i < num_layers_; i++){
        biases_[i - 1] = cv::Mat(sizes_[i], 1, CV_32FC1);
        // file the bias and weight with normally distributed random numbers.
        cv::randn(biases_[i - 1], cv::Scalar::all(0.0f), cv::Scalar::all(1.0f));
        weights_[i - 1] = cv::Mat(sizes_[i], sizes_[i - 1], CV_32FC1);
        cv::randn(weights_[i - 1], cv::Scalar::all(0.0f), cv::Scalar::all(1.0f));
    }
}

bool NeuralNetwork::FeedForward(const cv::Mat &input, cv::Mat &output) {
    if (input.rows != sizes_[0]) {
        LOG(ERROR) << "Input vector's length doesn't match with the net's input."
                   << std::endl;
        return false;
    }
    input.copyTo(output);
    for(int i = 1; i < num_layers_; i++) {
        output = sigmoid(weights_[i - 1] * output + biases_[i - 1]);
    }
    return true;
}

int NeuralNetwork::Predict(const cv::Mat &input) {
    cv::Mat y;
    if (!FeedForward(input, y)) {
        return -1;
    }
    cv::Point position;
    cv::minMaxLoc(y, nullptr, nullptr, nullptr, &position);
    return position.y;
}
bool NeuralNetwork::Verify(const cv::Mat &test_case) {
    cv::Mat x = test_case.colRange(0, test_case.cols - 1).t();
    return Predict(x) == static_cast<int>(test_case.at<float>(test_case.cols - 1));
}

void NeuralNetwork::DataNormalization(cv::Mat &data) {
    data.convertTo(data, CV_32F);   // convert the data from uchar to float
    data.colRange(0, data.cols - 1) /= 255; // the value from 0~255 to 0~1
}

bool NeuralNetwork::Train(const cv::Mat &training_samples,
                          int epochs, int mini_batch_size, float eta,
                          const cv::Mat &testing_samples) {

    cv::Mat training_data = training_samples.clone();
    cv::Mat testing_data = testing_samples.clone();

    // data pre-processing
    DataNormalization(training_data);
    if (!testing_data.empty()) {
        DataNormalization(testing_data);
    }
    // training and testing in every epoch
    for (int i = 0; i < epochs; i++) {
        RandomShuffle(training_data);
        for (int k = 0; k < training_data.rows; k = k + mini_batch_size) {
            int stop_row = std::min(k + mini_batch_size, training_data.rows);
            UpdateMiniBatch(training_data.rowRange(k, stop_row), eta);
        }
        if (!testing_data.empty()) {
            int precise_cout = 0;
            for (int k = 0; k < testing_data.rows; k++) {
                precise_cout += Verify(testing_data.row(k));
            }
            std::cout << "Epoch " << i << ": " << precise_cout
            << " / " << testing_data.rows << std::endl;
        } else {
            std::cout << "Epoch " << i << ": completed!" << std::endl;
        }
    }
    return true;
}

void NeuralNetwork::UpdateMiniBatch(const cv::Mat &subset_data, float eta) {
    int batch_size = subset_data.rows;
    std::vector<cv::Mat> nabla_w(num_layers_ - 1);
    std::vector<cv::Mat> nabla_b(num_layers_ - 1);
    std::vector<cv::Mat> delta_nabla_w(num_layers_ - 1);
    std::vector<cv::Mat> delta_nabla_b(num_layers_ - 1);

    for (int l = 0; l < num_layers_ - 1; l++) {
        nabla_w[l] = cv::Mat::zeros(weights_[l].size(), weights_[l].type());
        nabla_b[l] = cv::Mat::zeros(biases_[l].size(), biases_[l].type());
        delta_nabla_w[l] = cv::Mat::zeros(weights_[l].size(), weights_[l].type());
        delta_nabla_b[l] = cv::Mat::zeros(biases_[l].size(), biases_[l].type());
    }

    for (int i = 0; i < subset_data.rows; i++) {
        cv::Mat x = subset_data(cv::Range(i, i + 1), cv::Range(0, subset_data.cols - 1)).t();
        cv::Mat y(sizes_.back(), 1, CV_32F, cv::Scalar(0.0f));
        y.at<float>(static_cast<int>(subset_data.at<float>(i, subset_data.cols - 1))) = 1.0;
        BackPropagation(x, y, delta_nabla_w, delta_nabla_b);
        for (int l = 0; l < num_layers_ - 1; l++) {
            nabla_w[l] += delta_nabla_w[l];
            nabla_b[l] += delta_nabla_b[l];
        }
    }
    float lambda = 5;
    int totle_number = 60000;
    for (int l = 0; l < num_layers_ - 1; l++) {
        weights_[l] *= (1 - eta * lambda / totle_number);
        weights_[l] -= (eta / batch_size * nabla_w[l]);
        biases_[l] -= (eta / batch_size * nabla_b[l]);
    }
}


void NeuralNetwork::BackPropagation(const cv::Mat &x, const cv::Mat &y,
                                    std::vector<cv::Mat> &delta_nabla_w,
                                    std::vector<cv::Mat> &delta_nabla_b) {
    std::vector<cv::Mat> activations(num_layers_);
    std::vector<cv::Mat> weighted_input(num_layers_ - 1);
    activations[0] = x.clone();
    for(int i = 1; i < num_layers_; i++) {
        weighted_input[i - 1] = weights_[i - 1] * activations[i - 1] + biases_[i - 1];
        activations[i] = sigmoid(weighted_input[i - 1]);
    }
    // delta = (a- y) * g'(z)
    cv::Mat delta = (activations.back() - y);
    delta_nabla_b.back() = delta.clone();
    delta_nabla_w.back() = delta * activations[num_layers_ - 2].t();

    for (int l = num_layers_ - 2; l > 0; l--) {
        delta = (weights_[l].t() * delta).mul(sigmoid_primer(weighted_input[l - 1]));
        delta_nabla_b[l - 1] = delta.clone();
        delta_nabla_w[l - 1] = delta * activations[l - 1].t();
    }
}
