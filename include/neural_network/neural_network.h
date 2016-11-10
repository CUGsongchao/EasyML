//
// File name: neural_netWork.h
// Created by ronny on 16-7-7.
//

#ifndef EASYML_NEURALNETWORK_NEURALNETWORK_H
#define EASYML_NEURALNETWORK_NEURALNETWORK_H

#include <vector>
#include <initializer_list>  // std::initializer_list

#include <opencv2/core/core.hpp> // cv::Mat

class NeuralNetwork {
public:
    /// @breif      NeuralNetwork class constuctor
    /// @param[in]  sizes:contains the number of neurons in the respective layers.
    /// @return     NONE
    NeuralNetwork(const std::vector<unsigned int> &sizes);

    bool FeedForward(const cv::Mat &input, cv::Mat &output);

    int Predict(const cv::Mat &input);

    /// @brief      Train the neural network using mini-batch stochastic gradient descent.  
    /// @param[in]  trainging_samples[in] The "training_samples" is a Mat whose every row store 
    /// the (x1,x2,...,xn, y) representing the training inputs and the desired  outputs.  
    /// @param[in] epochs: the number of epochs to train for
    /// @param[in] mini_batch_size: the size of the mini-batches to use when sampling 
    /// @param[in] eta: learing rate
    /// @param[in] testing_sample  If "test_data" is provided then the network will be 
    /// evaluated against the test data after each epoch,
    /// and partial progress printed out. This is useful for tracking progress,
    /// but slows things down substantially
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


#endif // EASYML_NEURALNETWORK_NEURALNETWORK_H
