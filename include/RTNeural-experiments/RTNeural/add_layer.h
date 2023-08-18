#pragma once

#include <RTNeural/RTNeural.h>

#if RTNEURAL_USE_EIGEN
namespace RTNeural
{
namespace experimental
{
    template <typename T, int size>
    class Add
    {
        using vec_type = Eigen::Matrix<T, size, 1>;

    public:
        static constexpr auto in_size = size;
        static constexpr auto out_size = size;

        Add()
            : outs(outs_internal)
        {
        }

        inline void forward(const Eigen::Matrix<T, in_size, 1>& ins1, const Eigen::Matrix<T, in_size, 1>& ins2) noexcept
        {
            outs = ins1 + ins2;
        }

        Eigen::Map<vec_type, RTNeuralEigenAlignment> outs;

    private:
        T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
    };
}
}
#elif RTNEURAL_USE_XSIMD
namespace RTNeural
{
namespace experimental
{
    template <typename T, int size>
    class Add
    {
        using v_type = xsimd::simd_type<T>;
        static constexpr auto v_size = (int)v_type::size;
        static constexpr auto v_io_size = ceil_div(size, v_size);

    public:
        static constexpr auto in_size = size;
        static constexpr auto out_size = size;

        Add() = default;

        inline void forward(const v_type (&ins1)[v_io_size], const v_type (&ins2)[v_io_size]) noexcept
        {
            for(int i = 0; i < v_io_size; ++i)
                outs[i] = ins1[i] + ins2[i];
        }

        v_type outs[v_io_size];
    };
}
}
#else // RTNEURAL_USE_STL
namespace RTNeural
{
namespace experimental
{
    template <typename T, int size>
    class Add
    {
    public:
        static constexpr auto in_size = size;
        static constexpr auto out_size = size;

        Add() = default;

        inline void forward(const T (&ins1)[size], const T (&ins2)[size]) noexcept
        {
            for(int i = 0; i < size; ++i)
                outs[i] = ins1[i] + ins2[i];
        }

        T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[size];
    };
}
}
#endif
