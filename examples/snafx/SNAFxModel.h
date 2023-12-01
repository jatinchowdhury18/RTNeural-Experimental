#pragma once

#include "TCNBlock.h"

namespace snafx
{
template <typename T, int n_channels = 32, int kernel_size = 9, int cond_dim = 2>
struct Model
{
    void load_model (const nlohmann::json& model_json)
    {
        block0.load_weights (model_json, 0);
        block1.load_weights (model_json, 1);
        block2.load_weights (model_json, 2);
        block3.load_weights (model_json, 3);
    }

    void reset()
    {
        block0.reset();
        block1.reset();
        block2.reset();
        block3.reset();
    }

    void condition (const T (&cond_ins)[cond_dim]) noexcept
    {
        block0.film.condition (cond_ins);
        block1.film.condition (cond_ins);
        block2.film.condition (cond_ins);
        block3.film.condition (cond_ins);
    }

    T forward (T input) noexcept
    {
#if RTNEURAL_USE_XSIMD
        std::fill (std::begin (arr), std::end (arr), T{});
        arr[0] = input;
        block0.forward ({ xsimd::load_aligned (arr) });
#elif RTNEURAL_USE_EIGEN
        ins (0) = input;
        block0.forward (ins);
#endif
        block1.forward (block0.outs);
        block2.forward (block1.outs);
        block3.forward (block2.outs);

#if RTNEURAL_USE_XSIMD
        block3.outs[0].store_aligned (arr);
        return arr[0];
#elif RTNEURAL_USE_EIGEN
        return block3.outs (0);
#endif
    }

    TCNBlock<T, 1, n_channels, cond_dim, kernel_size, 1> block0;
    TCNBlock<T, n_channels, n_channels, cond_dim, kernel_size, 10> block1;
    TCNBlock<T, n_channels, n_channels, cond_dim, kernel_size, 100> block2;
    TCNBlock<T, n_channels, 1, cond_dim, kernel_size, 1000> block3;

private:
#if RTNEURAL_USE_XSIMD
    alignas (RTNEURAL_DEFAULT_ALIGNMENT) T arr[xsimd::batch<T>::size] {};
#elif RTNEURAL_USE_EIGEN
    Eigen::Vector<T, 1> ins {};
#endif
};
}
