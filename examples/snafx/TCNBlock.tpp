namespace snafx
{
void reverse_kernels(std::vector<std::vector<std::vector<float>>>& conv_weights)
{
    for (auto& channel_weights : conv_weights)
    {
        for (auto& kernel : channel_weights)
        {
            std::reverse(kernel.begin(), kernel.end());
        }
    }
}

template <typename T, int in_size, int out_size, int cond_dim, int kernel_size, int dilation_rate>
void TCNBlock<T, in_size, out_size, cond_dim, kernel_size, dilation_rate>::load_weights (const nlohmann::json& model_json, int block_index)
{
    const auto conv_weights_key = std::string { "blocks." } + std::to_string (block_index) + ".conv.weight";
    std::vector<std::vector<std::vector<T>>> conv_weights = model_json.at (conv_weights_key);
    reverse_kernels (conv_weights);
    conv.setWeights (conv_weights);

    const auto conv_bias_key = std::string { "blocks." } + std::to_string (block_index) + ".conv.bias";
    std::vector<T> conv_bias = model_json.at (conv_bias_key);
    conv.setBias (conv_bias);

    film.load_weights (model_json, block_index);

    const auto act_weights_key = std::string { "blocks." } + std::to_string (block_index) + ".act.weight";
    const auto act_weight = model_json.at (act_weights_key).at (0).get<T>();
    act.setAlphaVals ({ act_weight });

    const auto res_weights_key = std::string { "blocks." } + std::to_string (block_index) + ".res.weight";
    std::vector<std::vector<std::vector<T>>> res_weights = model_json.at (res_weights_key);
    reverse_kernels (res_weights);
    res.setWeights (res_weights);

    std::vector<T> res_bias (res.out_size, (T) 0);
    res.setBias (res_bias);
}

template <typename T, int in_size, int out_size, int cond_dim, int kernel_size, int dilation_rate>
void TCNBlock<T, in_size, out_size, cond_dim, kernel_size, dilation_rate>::reset()
{
    conv.reset();
    res.reset();
}
} // namespace snafx
