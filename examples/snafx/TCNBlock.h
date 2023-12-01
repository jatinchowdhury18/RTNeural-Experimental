#pragma once

#include "FiLM.h"

namespace snafx
{
template <typename T, int in_size, int out_size, int cond_dim, int kernel_size, int dilation_rate>
struct TCNBlock
{
#if RTNEURAL_USE_XSIMD
    using v_type = xsimd::batch<T>;
    static constexpr auto v_size = (int) v_type::size;
    static constexpr auto in_size_v = RTNeural::ceil_div (in_size, v_size);
    static constexpr auto out_size_v = RTNeural::ceil_div (out_size, v_size);
#endif

    void load_weights (const nlohmann::json& model_json, int block_index);

    void reset();

#if RTNEURAL_USE_XSIMD
    inline void forward (const v_type (&ins)[in_size_v]) noexcept
#elif RTNEURAL_USE_EIGEN
    inline void forward (const Eigen::Vector<T, in_size>& ins) noexcept
#endif
    {
        conv.forward (ins);
        film.forward (conv.outs);
        act.forward (film.outs);

        res.forward (ins);

#if RTNEURAL_USE_XSIMD
        for (int i = 0; i < out_size_v; ++i)
            outs[i] = act.outs[i] + res.outs[i];
#elif RTNEURAL_USE_EIGEN
        outs = act.outs + res.outs;
#endif
    }

    RTNeural::Conv1DT<T, in_size, out_size, kernel_size, dilation_rate, true> conv;
    FiLM<T, out_size, cond_dim> film;
    RTNeural::PReLUActivationT<T, out_size> act;
    RTNeural::Conv1DT<T, in_size, out_size, 1, 1> res;

#if RTNEURAL_USE_XSIMD
    v_type outs[out_size_v]{};
#elif RTNEURAL_USE_EIGEN
    Eigen::Vector<T, out_size> outs {};
#endif
};
}

#include "TCNBlock.tpp"
