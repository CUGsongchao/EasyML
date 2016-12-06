#include <iostream>
#include <cstdlib>
#include <glog/logging.h>

#include <memory>

#include "neural_network/net.h"
#include "neural_network/input_layer.h"

#include "neural_network/fully_connected_layer.h"
#include "neural_network/output_layer.h"
#include <util/util.h>


int main(int argc, char *argv[]) {
    using namespace easyml;
    using namespace nn;
    // set glog configure
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold=google::INFO;
    FLAGS_colorlogtostderr=true;


    cv::Mat training_data;
    cv::Mat testing_data;
    cv::Mat labels;

    // TODO splite the training data to x and lable
    if (!util::LoadMNIST("../data/", training_data, testing_data)) {
        LOG(ERROR) << "Failed to load mnist data" << std::endl;
        exit(EXIT_FAILURE);
    }

    // initilize the net
    std::unique_ptr<nn::Net> net;
    InputLayerParameter input_param("input");
    std::shared_ptr<nn::Layer> input_layer = std::make_shared<nn::InputLayer>(input_param);
    net->PushBack(input_layer);

    FullyConnectedLayerParameter hidden_param(
            "hidden",
            Dim(1, 1, 784, 1),
            Dim(1, 1, 50, 1),
            std::shared_ptr<util::SigmoidFunction>(new util::SigmoidFunction())
    );
    std::shared_ptr<nn::Layer> hidden = std::make_shared<nn::FullyConnectedLayer>(hidden_param);
    net->PushBack(hidden);


    OutputLayerParameter output_param(
            "output",
            Dim(1, 1, 50, 1),
            Dim(1, 1, 10, 1),
            std::shared_ptr<util::SigmoidFunction>(new util::SigmoidFunction()),
            std::shared_ptr<util::CEEFunction>(new util::CEEFunction())
    );
    std::shared_ptr<nn::Layer> output_layer 
        = std::make_shared<nn::OutputLayer>(output_param, labels);
    net->PushBack(output_layer);


    

    LOG(INFO) << "Start to train" << std::endl;

    nn::NNTrainParam param;

    net->Train(training_data, labels, param);
    LOG(INFO) << "Complete to train" << std::endl;

    return 0;
}
