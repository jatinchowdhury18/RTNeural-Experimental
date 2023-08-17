#pragma once

#include <Accelerate/Accelerate.h>
#include <RTNeural/RTNeural.h>

namespace RTNeural::accelerate
{
static inline void sigmoid(const float* in, float* out, int dim) noexcept
{
    constexpr float one = 1.0f;
    constexpr float neg_one = -1.0f;
    const auto dim_int = static_cast<int>(dim);

    vDSP_vsmul(in, 1, &neg_one, out, 1, dim);
    vvexpf(out, out, &dim_int);
    vDSP_vsadd(out, 1, &one, out, 1, dim);
    vvrecf(out, out, &dim_int);
}

static inline void sigmoid(const double* in, double* out, int dim) noexcept
{
    constexpr double one = 1.0;
    constexpr double neg_one = -1.0;
    const auto dim_int = static_cast<int>(dim);

    vDSP_vsmulD(in, 1, &neg_one, out, 1, dim);
    vvexp(out, out, &dim_int);
    vDSP_vsaddD(out, 1, &one, out, 1, dim);
    vvrec(out, out, &dim_int);
}

static inline void softmax(const float* in, float* out, int dim) noexcept
{
    const auto dim_int = static_cast<int>(dim);
    float exp_sum;

    vvexpf(out, in, &dim_int);
    vDSP_sve(out, 1, &exp_sum, dim);
    vDSP_vsdiv(out, 1, &exp_sum, out, 1, dim);
}

static inline void softmax(const double* in, double* out, int dim) noexcept
{
    const auto dim_int = static_cast<int>(dim);
    double exp_sum;

    vvexp(out, in, &dim_int);
    vDSP_sveD(out, 1, &exp_sum, dim);
    vDSP_vsdivD(out, 1, &exp_sum, out, 1, dim);
}
}
