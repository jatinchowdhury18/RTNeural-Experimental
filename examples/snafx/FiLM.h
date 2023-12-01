#pragma once

#include <RTNeural/RTNeural.h>

namespace snafx
{
template <typename T, int num_features, int cond_dim, bool use_batch_norm = false>
struct FiLM
{
#if RTNEURAL_USE_XSIMD
    using v_type = xsimd::batch<T>;
    static constexpr auto v_size = (int) v_type::size;
    static_assert (num_features == 1 || num_features % v_size == 0, "This implementation relies on num_features being an even multiple of batch_size!");
    static constexpr auto num_features_vec = RTNeural::ceil_div (num_features, v_size);
#endif

    void load_weights (const nlohmann::json& model_json, int block_index);

    void condition (const T (&cond_ins)[cond_dim]) noexcept
    {
#if RTNEURAL_USE_XSIMD
        std::copy (std::begin (cond_ins), std::end (cond_ins), reinterpret_cast<T*> (cond_in));
        adaptor.forward (cond_in);
        std::copy (adaptor.outs, adaptor.outs + num_features_vec, g_vals);
        std::copy (adaptor.outs + num_features_vec, adaptor.outs + 2 * num_features_vec, b_vals);
#elif RTNEURAL_USE_EIGEN
        std::copy (std::begin (cond_ins), std::end (cond_ins), cond_in.data());
        adaptor.forward (cond_in);
        std::copy (adaptor.outs.data(), adaptor.outs.data() + num_features, g_vals.data());
        std::copy (adaptor.outs.data() + num_features, adaptor.outs.data() + 2 * num_features, b_vals.data());
#endif
    }

#if RTNEURAL_USE_XSIMD
    template <bool B = use_batch_norm>
    inline typename std::enable_if<B, void>::type
        forward (const v_type (&ins)[num_features_vec]) noexcept
    {
        bn.forward (ins);
        forward<false> (bn.outs);
    }

    template <bool B = use_batch_norm>
    inline typename std::enable_if<! B, void>::type
        forward (const v_type (&ins)[num_features_vec]) noexcept
    {
        for (int i = 0; i < num_features_vec; ++i)
            outs[i] = g_vals[i] * ins[i] + b_vals[i];
    }
#elif RTNEURAL_USE_EIGEN
    template <bool B = use_batch_norm>
    inline typename std::enable_if<B, void>::type
        forward (const Eigen::Vector<T, num_features>& ins) noexcept
    {
        bn.forward (ins);
        forward<false> (bn.outs);
    }

    template <bool B = use_batch_norm>
    inline typename std::enable_if<! B, void>::type
        forward (const Eigen::Vector<T, num_features>& ins) noexcept
    {
        outs = g_vals.cwiseProduct (ins) + b_vals;
    }
#endif

    RTNeural::DenseT<T, cond_dim, 2 * num_features> adaptor;
    RTNeural::BatchNorm1DT<T, num_features, false> bn;

#if RTNEURAL_USE_XSIMD
    v_type outs[num_features_vec] {};
#elif RTNEURAL_USE_EIGEN
    Eigen::Vector<T, num_features> outs {};
#endif

private:
#if RTNEURAL_USE_XSIMD
    v_type cond_in[RTNeural::ceil_div (cond_dim, v_size)] {};
    v_type g_vals[num_features_vec] {};
    v_type b_vals[num_features_vec] {};
#elif RTNEURAL_USE_EIGEN
    Eigen::Vector<T, cond_dim> cond_in {};
    Eigen::Vector<T, num_features> g_vals {};
    Eigen::Vector<T, num_features> b_vals {};
#endif
};
} // namespace snafx

#include "FiLM.tpp"
