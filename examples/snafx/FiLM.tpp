namespace snafx
{
template <typename T, int num_features, int cond_dim, bool use_batch_norm>
void FiLM<T, num_features, cond_dim, use_batch_norm>::load_weights (const nlohmann::json& model_json, int block_index)
{
    const auto adaptor_weights_key = std::string { "blocks." } + std::to_string (block_index) + ".film.adaptor.weight";
    std::vector<std::vector<T>> adaptor_weights = model_json.at (adaptor_weights_key);
    adaptor.setWeights (adaptor_weights);

    const auto adaptor_bias_key = std::string { "blocks." } + std::to_string (block_index) + ".film.adaptor.bias";
    std::vector<T> adaptor_bias = model_json.at (adaptor_bias_key).get<std::vector<T>>();
    adaptor.setBias(adaptor_bias.data());

    static_assert (! use_batch_norm, "TODO: figure out how to load the batch norm weights once we have a test model!");
}
} // namespace snafx
