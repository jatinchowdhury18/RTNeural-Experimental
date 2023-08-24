#include <RTNeural/RTNeural.h>

template <typename T, typename TQuant, typename DenseType, typename Quantizer>
void loadDenseQuantized(const nlohmann::json& modelJson, const std::string& layerPrefix,
    DenseType& dense, Quantizer&& quantizer, bool hasBias = true)
{
    std::vector<std::vector<T>> dense_weights = modelJson.at(layerPrefix + "weight");
    std::vector<std::vector<TQuant>> dense_weights_quant = modelJson.at(layerPrefix + "weight");
    for(size_t i = 0; i < dense_weights.size(); ++i)
        for(size_t j = 0; j < dense_weights[0].size(); ++j)
            dense_weights_quant[i][j] = quantizer(dense_weights[i][j]);
    dense.setWeights(dense_weights_quant);

    if(hasBias)
    {
        //        const std::vector<T> dense_bias = modelJson.at(layerPrefix + "bias");
        //        dense.setBias(dense_bias.data());
    }
    else
    {
        const std::vector<TQuant> dense_bias((size_t)dense.out_size, (TQuant)0);
        dense.setBias(dense_bias.data());
    }
}

template <typename out_type, typename in_type>
out_type quantize(in_type x, in_type scale, in_type zero_point)
{
    return static_cast<out_type>(std::round((x / scale) + zero_point));
}

template <typename out_type, typename in_type>
out_type unquantize(in_type x, out_type scale, out_type zero_point)
{
    return (static_cast<out_type>(x) - zero_point) * scale;
}

int main()
{
    std::cout << "Running \"quantization\" example..." << std::endl;

    const auto model_file = std::string { RTNEURAL_EXPERIMENTS_SOURCE_DIR } + std::string { "/examples/quantization/model_fp32.json" };
    std::cout << "Loading model from path: " << model_file << std::endl;
    std::ifstream json_stream { model_file, std::ifstream::binary };
    nlohmann::json model_json;
    json_stream >> model_json;

    std::cout << "Running fp32 model\n";
    RTNeural::DenseT<float, 4, 4> dense1;
    RTNeural::torch_helpers::loadDense<float>(model_json, "dense1.", dense1, false);

    static constexpr float input_data[4][4] {
        { +1.0f, +2.0f, +3.0f, +4.0f },
        { +3.0f, +4.0f, +5.0f, +6.0f },
        { -1.0f, -2.0f, -3.0f, -4.0f },
        { +1.1f, +2.2f, +3.3f, +4.4f },
    };

    for(auto& arr : input_data)
    {
        dense1.forward(arr);
        std::cout << "[" << dense1.outs[0] << ", "
                  << dense1.outs[1] << ", "
                  << dense1.outs[2] << ", "
                  << dense1.outs[3] << "]" << std::endl;
    }

    //    const auto q = quantize<int16_t> (1.0f, 0.0392f, 102.0f);
    //    const auto tester = unquantize (q, 0.0392f, 102.0f);

    std::cout << "Running int8 model\n";
    static constexpr auto model_scale = 1.0e-2f;
    static constexpr auto model_zero_point = 0.0f;
    static constexpr auto dense_scale = 1.0e-4f;
    static constexpr auto dense_zero_point = 0.0f;

    RTNeural::DenseT<int32_t, 4, 4> dense1_q;
    loadDenseQuantized<float, int32_t>(
        model_json,
        "dense1.", dense1_q, [](auto x)
        { return quantize<int32_t>(x, dense_scale, dense_zero_point); },
        false);

    for(auto& arr : input_data)
    {
        alignas(16) int32_t input_q[4] {};
        for(int i = 0; i < 4; ++i)
            input_q[i] = quantize<int32_t>(arr[i], model_scale, model_zero_point);

        //        std::cout << "[" << arr[0] << ", "
        //                  << arr[1] << ", "
        //                  << arr[2] << ", "
        //                  << arr[3] << "]" << std::endl;
        //        std::cout << "[" << (int)input_q[0] << ", "
        //                  << (int)input_q[1] << ", "
        //                  << (int)input_q[2] << ", "
        //                  << (int)input_q[3] << "]" << std::endl;
        //        std::cout << "[" << unquantize(input_q[0], model_scale, model_zero_point) << ", "
        //                  << unquantize(input_q[1], model_scale, model_zero_point) << ", "
        //                  << unquantize(input_q[2], model_scale, model_zero_point) << ", "
        //                  << unquantize(input_q[3], model_scale, model_zero_point) << "]" << std::endl;

        dense1_q.forward(input_q);

        //        std::cout << "[" << (int)dense1_q.outs[0] << ", "
        //                  << (int)dense1_q.outs[1] << ", "
        //                  << (int)dense1_q.outs[2] << ", "
        //                  << (int)dense1_q.outs[3] << "]" << std::endl;
        std::cout << "[" << unquantize(dense1_q.outs[0], model_scale * dense_scale, model_zero_point + dense_zero_point) << ", "
                  << unquantize(dense1_q.outs[1], model_scale * dense_scale, model_zero_point + dense_zero_point) << ", "
                  << unquantize(dense1_q.outs[2], model_scale * dense_scale, model_zero_point + dense_zero_point) << ", "
                  << unquantize(dense1_q.outs[3], model_scale * dense_scale, model_zero_point + dense_zero_point) << "]" << std::endl;
    }

    return 0;
}

// Torch output (un-quantized)
// [[-2.1737456   1.6316392  -0.62620986 -1.6842039 ]
//  [-3.203773    2.2877185  -0.94911987 -3.6770139 ]
//  [ 2.1737456  -1.6316392   0.62620986  1.6842039 ]
//  [-2.3911202   1.7948034  -0.68883085 -1.8526242 ]]

// Torch output (quantized)
// [[-2.1743107  1.636578  -0.6312515 -1.6833373]
//  [-3.203017   2.2912092 -0.9585671 -3.6706107]
//  [ 2.1743107 -1.636578   0.6312515  1.6833373]
//  [-2.384728   1.8002357 -0.6780109 -1.8469951]]
