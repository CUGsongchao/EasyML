#include <iostream>
#include <cstdlib>
#include <glog/logging.h>

#include "neural_network/neural_network.h"
#include "util/util.h"

int main(int argc, char *argv[]) {
    // set glog configure
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold=google::INFO;
    FLAGS_colorlogtostderr=true;

    // initilize the net
    std::vector<unsigned> layer_param = {784, 50, 10};
    NeuralNetwork net(layer_param);

    cv::Mat training_data, testing_data;
    if (!Util::LoadMNIST("../data/", training_data, testing_data)) {
        LOG(ERROR) << "Failed to load mnist data" << std::endl;
        exit(EXIT_FAILURE);
    }
    LOG(INFO) << "Start to train" << std::endl;
    net.Train(training_data, 30, 10, 0.5f, testing_data);
    LOG(INFO) << "Complete to train" << std::endl;

    return 0;
}
