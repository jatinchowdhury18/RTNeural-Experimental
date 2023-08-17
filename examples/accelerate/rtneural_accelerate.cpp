#include <iostream>

#include <RTNeural/RTNeural.h>

#include <RTNeural/activation_accelerate.h>
#include <RTNeural/dense_accelerate.h>

std::string getModelFile()
{
    return std::string { RTNEURAL_SOURCE_DIR }
    + std::string { "/examples/rtneural_static_model/test_net.json" };
}

int main()
{
    std::cout << "Running \"rtneural static model\" example..." << std::endl;

    auto modelFilePath = getModelFile();
    std::cout << "Loading model from path: " << modelFilePath << std::endl;
    std::ifstream jsonStream(modelFilePath, std::ifstream::binary);
    nlohmann::json modelJson;
    jsonStream >> modelJson;

    RTNeural::accelerate::Dense<float> dense_1 { 1, 8 };
    RTNeural::accelerate::TanhActivation<float> tanh_1 { 8 };
    RTNeural::accelerate::Dense<float> dense_2 { 8, 8 };
    RTNeural::accelerate::ReLuActivation<float> relu_2 { 8 };
    RTNeural::accelerate::Dense<float> dense_3 { 8, 8 };
    RTNeural::accelerate::SoftmaxActivation<float> softmax_3 { 8 };
    RTNeural::accelerate::Dense<float> dense_out { 8, 1 };

    RTNeural::json_parser::loadDense<float>(dense_1, modelJson.at("layers").at(0).at("weights"));
    RTNeural::json_parser::loadDense<float>(dense_2, modelJson.at("layers").at(1).at("weights"));
    RTNeural::json_parser::loadDense<float>(dense_3, modelJson.at("layers").at(2).at("weights"));
    RTNeural::json_parser::loadDense<float>(dense_out, modelJson.at("layers").at(3).at("weights"));

    std::array<float, 8> data {};
    data[0] = 5.0f;

    dense_1.forward(data.data(), data.data());
    tanh_1.forward(data.data(), data.data());
    dense_2.forward(data.data(), data.data());
    relu_2.forward(data.data(), data.data());
    dense_3.forward(data.data(), data.data());
    softmax_3.forward(data.data(), data.data());
    dense_out.forward(data.data(), data.data());

    std::cout << "Test output: " << data[0] << std::endl;

    return 0;
}
