#include <easyml/neural_network/output_layer.h>
#include <glog/logging.h>

namespace easyml {
namespace nn {



OutputLayer::OutputLayer(const OutputLayerParameter &param, const cv::Mat &labels)
{
    name_ = param.name;
    type_ = param.type;
    cost_ = param.cost_function;
    activation_ = param.activation;

    biases_ = cv::Mat(param.output_dim.height, 1, CV_32FC1);
    cv::randn(biases_, cv::Scalar::all(0.0f), cv::Scalar::all(1.0f));

    weights_ = cv::Mat(param.output_dim.height, param.input_dim.height, CV_32FC1);
    float init_weight = 1.0f / sqrt(static_cast<float>(param.input_dim.height));
    cv::randn(weights_, cv::Scalar::all(0.0f), cv::Scalar::all(init_weight));

    input_.assign(param.input_dim.batch_size,
            cv::Mat(param.input_dim.height, param.input_dim.width, CV_32FC1));

    for (int i = 0; i < labels.rows; i++) {
        labels_.push_back(labels.row(i).t());
    }
}


void OutputLayer::FeedForward(
        const std::vector<cv::Mat> &input,
        std::vector<cv::Mat> &output)
{
    CHECK_EQ(input_.size(), input.size()) 
        << "batch size of the input doesn't match with the net.";

    int batch_size = input_.size();
    output.assign(batch_size, cv::Mat());
    weighted_output_.assign(batch_size, cv::Mat());

    for (int i = 0; i < batch_size; i++) {
        input_[i] = input[i].clone();
        weighted_output_[i] = weights_ * input_[i] + biases_;
        output[i] = (*activation_)(weighted_output_[i]);
    }
}

void OutputLayer::BackPropagation(
        const std::vector<cv::Mat> &delta_in,
        std::vector<cv::Mat> &delta_out,
        float eta,
        float lambda)
{
    cv::Mat nabla_w_sum(weights_.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::Mat nabla_b_sum(biases_.size(), CV_32FC1, cv::Scalar(0.0f));

    int batch_size = input_.size();

    for (int i = 0; i < batch_size; i++) {
        delta_out[i] = cost_->CostDerivation((*activation_)(weighted_output_[i]), labels_[i])
            .mul(activation_->primer(weighted_output_[i]));
        cv::Mat nabla_b = delta_out[i].clone();
        cv::Mat nabla_w = nabla_b * input_[i];
        nabla_w_sum += nabla_w;
        nabla_b_sum += nabla_b;
    }
    weights_ *= (1 - eta * lambda); // L2 regularization
    weights_ -= (eta / batch_size) * nabla_w_sum;
    biases_ -= (eta / batch_size) * nabla_b_sum;
}

} // namespace easyml
} // namespace nn


