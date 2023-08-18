#include "tests/load_csv.hpp"
#include <RTNeural/RTNeural.h>
#include <RTNeural/add_layer.h>

int main()
{
    std::cout << "Running \"residual connection\" example..." << std::endl;

    const auto model_file = std::string { RTNEURAL_EXPERIMENTS_SOURCE_DIR } + std::string { "/examples/residual_connection/res_connection.json" };

    std::cout << "Loading model from path: " << model_file << std::endl;
    std::ifstream json_stream { model_file, std::ifstream::binary };
    nlohmann::json model_json;
    json_stream >> model_json;

    const auto input_file = std::string { RTNEURAL_EXPERIMENTS_SOURCE_DIR } + std::string { "/examples/residual_connection/res_connection_x_python.csv" };
    std::ifstream model_inputs_file { input_file };
    std::vector<float> reference_inputs = load_csv::loadFile<float>(model_inputs_file);

    const auto output_file = std::string { RTNEURAL_EXPERIMENTS_SOURCE_DIR } + std::string { "/examples/residual_connection/res_connection_y_python.csv" };
    std::ifstream model_outputs_file { output_file };
    std::vector<float> reference_outputs = load_csv::loadFile<float>(model_outputs_file);

    RTNeural::DenseT<float, 1, 8> dense1 {};
    RTNeural::TanhActivationT<float, 8> tanh1 {};
    RTNeural::DenseT<float, 8, 8> dense2 {};
    RTNeural::TanhActivationT<float, 8> tanh2 {};
    RTNeural::experimental::Add<float, 8> res {};
    RTNeural::DenseT<float, 8, 1> dense_out {};

    RTNeural::json_parser::loadDense<float>(dense1, model_json.at("layers").at(0).at("weights"));
    RTNeural::json_parser::loadDense<float>(dense2, model_json.at("layers").at(1).at("weights"));
    RTNeural::json_parser::loadDense<float>(dense_out, model_json.at("layers").at(3).at("weights"));

#if RTNEURAL_USE_EIGEN
    using InputType = Eigen::Matrix<float, 1, 1>;
#elif RTNEURAL_USE_XSIMD
    using InputType = xsimd::batch<float>[1];
#else
    using InputType = float[1];
#endif

    std::vector<float> test_outputs;
    test_outputs.resize(reference_inputs.size());
    for(size_t i = 0; i < reference_inputs.size(); ++i)
    {
        dense1.forward(InputType { reference_inputs[i] });
        tanh1.forward(dense1.outs);
        dense2.forward(tanh1.outs);
        tanh2.forward(dense2.outs);
        res.forward(tanh2.outs, tanh1.outs);
        dense_out.forward(res.outs);
#if RTNEURAL_USE_XSIMD
        test_outputs[i] = dense_out.outs[0].get (0);
#else
        test_outputs[i] = dense_out.outs[0];
#endif
    }

    for(size_t i = 0; i < reference_inputs.size(); ++i)
        std::cout << reference_outputs[i] << " | " << test_outputs[i] << std::endl;

    return 0;
}
