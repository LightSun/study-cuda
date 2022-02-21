#ifdef LIB_TORCH
#include <iostream>
#include <memory>
#include <vector>
#include "torch/torch.h"

struct Net : torch::nn::Module {
    Net(int input_size, int layer_width) {
        linear1 = register_module("linear1",
                                  torch::nn::Linear(input_size, layer_width));
        linear2 = register_module("linear2",
                                  torch::nn::Linear(layer_width, layer_width));
        linear3 = register_module("linear3",
                                  torch::nn::Linear(layer_width, layer_width));
        // set weights
        std::vector<float> weights = {0.1, 0.2, 0.3};
        std::vector<float> biases = {0.4, 0.4, 0.4};
        // layer 1
        linear1->bias[0] = biases[0];
        linear1->bias[1] = biases[1];
        linear1->bias[2] = biases[2];
        linear1->weight[0] = torch::from_blob(weights.data(),
                                              layer_width).clone();
        linear1->weight[1] = torch::from_blob(weights.data(),
                                              layer_width).clone();
        linear1->weight[2] = torch::from_blob(weights.data(),
                                              layer_width).clone();
        // layer 2
        linear2->bias[0] = biases[0];
        linear2->bias[1] = biases[1];
        linear2->bias[2] = biases[2];
        linear2->bias = torch::from_blob(biases.data(),
                                         layer_width).clone();
        linear2->weight[0] = torch::from_blob(weights.data(),
                                              layer_width).clone();
        linear2->weight[1] = torch::from_blob(weights.data(),
                                              layer_width).clone();
        linear2->weight[2] = torch::from_blob(weights.data(),
                                              layer_width).clone();
        // layer 3
        linear3->bias[0] = biases[0];
        linear3->bias[1] = biases[1];
        linear3->bias[2] = biases[2];
        linear3->bias = torch::from_blob(biases.data(),
                                         layer_width).clone();
        linear3->weight[0] = torch::from_blob(weights.data(),
                                              layer_width).clone();
        linear3->weight[1] = torch::from_blob(weights.data(),
                                              layer_width).clone();
        linear3->weight[2] = torch::from_blob(weights.data(),
                                              layer_width).clone();
    }

    torch::Tensor forward(torch::Tensor input) {
        torch::Tensor output;
        output = torch::relu(linear1(input));
        output = torch::relu(linear2(output));
        output = linear3(output);
        return output;
    }

    torch::nn::Linear linear1{nullptr};
    torch::nn::Linear linear2{nullptr};
    torch::nn::Linear linear3{nullptr};
};

int main_libtorch() {
    int input_size = 3;
    int layer_width = 3;
    torch::autograd::GradMode::set_enabled(false);
    std::shared_ptr<Net> net = std::make_shared<Net>(input_size, layer_width);

    std::vector<float> before = {0.1, 0.2, 0.3};
    torch::Tensor a = torch::from_blob(before.data(), 3).clone();
    for (int i = 0; i < 100000; i++) {
        net->forward(a);
    }
    std::cout << "Torch output:\n";
    std::cout << net->forward(a) << std::endl;

    std::vector<float> after(3);
    std::vector<float> weights = {0.1, 0.2, 0.3};
    float bias = 0.4;
    for (int i = 0; i < 100000; i++) {
        before = {0.1, 0.2, 0.3};
        for (int layer = 0; layer < 3; layer++) {
            for (int node = 0; node < 3; node++) {
                after[node] = 0.0;
                for (int weight_idx = 0; weight_idx < 3; weight_idx++) {
                    after[node] += before[weight_idx] * weights[weight_idx];
                }
                after[node] += bias;
                after[node] = std::max(0.0f, after[node]);
            }
            before = after;
        }
    }
    std::cout << "My output:\n";
    for (auto i : after) std::cout << i << "\n";
    return 0;
}
#endif
