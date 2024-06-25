/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*! \file
 *  \brief gbyte.hpp provides data transfer counts of Sparse Linear Algebra Subprograms
 *  of Level 1, 2 and 3.
 */

#pragma once
#ifndef GBYTE_HPP
#define GBYTE_HPP

// Compute gbytes
inline double get_gpu_gbyte(double gpu_time_used, double gbyte_count)
{
    return gbyte_count / gpu_time_used * 1e6;
}

template <typename F, typename... Ts>
inline double get_gpu_gbyte(double gpu_time_used, F count, Ts... ts)
{
    return get_gpu_gbyte(gpu_time_used, count(ts...));
}

inline double get_gpu_time_msec(double gpu_time_used)
{
    return gpu_time_used / 1e3;
}

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
template <typename T, typename I>
constexpr double axpby_gbyte_count(I nnz)
{
    return (nnz * sizeof(I) + (3.0 * nnz) * sizeof(T)) / 1e9;
}

template <typename X, typename Y, typename I>
constexpr double doti_gbyte_count(I nnz)
{
    return (nnz * sizeof(I) + nnz * (sizeof(X) + sizeof(Y))) / 1e9;
}

template <typename T, typename I>
constexpr double gthr_gbyte_count(I nnz)
{
    return (nnz * sizeof(I) + (2.0 * nnz) * sizeof(T)) / 1e9;
}

template <typename T, typename I>
constexpr double gthrz_gbyte_count(I nnz)
{
    return (nnz * sizeof(I) + (2.0 * nnz) * sizeof(T)) / 1e9;
}

template <typename T, typename I>
constexpr double roti_gbyte_count(I nnz)
{
    return (nnz * sizeof(I) + (3.0 * nnz) * sizeof(T)) / 1e9;
}

template <typename T, typename I>
constexpr double sctr_gbyte_count(I nnz)
{
    return (nnz * sizeof(I) + (2.0 * nnz) * sizeof(T)) / 1e9;
}

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template <typename A, typename X, typename Y, typename I, typename J>
constexpr double bsrmv_gbyte_count(J mb, J nb, I nnzb, J block_dim, bool beta = false)
{
    return (sizeof(I) * (mb + 1) + sizeof(J) * nnzb + sizeof(A) * nnzb * block_dim * block_dim
            + sizeof(Y) * (mb * block_dim + (beta ? mb * block_dim : 0))
            + sizeof(X) * (nb * block_dim))
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double bsrmv_gbyte_count(J mb, J nb, I nnzb, J block_dim, bool beta = false)
{
    return bsrmv_gbyte_count<T, T, T>(mb, nb, nnzb, block_dim, beta);
}

template <typename T>
constexpr double bsrsv_gbyte_count(int mb, int nnzb, int bsr_dim)
{
    return ((mb + 1 + nnzb) * sizeof(int)
            + (bsr_dim * (mb + mb + nnzb * bsr_dim)) * sizeof(T))
           / 1e9;
}

template <typename A, typename X, typename Y, typename I>
constexpr double coomv_gbyte_count(I M, I N, int64_t nnz, bool beta = false)
{
    return (sizeof(I) * 2.0 * nnz + sizeof(A) * nnz + sizeof(Y) * (M + (beta ? M : 0))
            + sizeof(X) * N)
           / 1e9;
}

template <typename T, typename I>
constexpr double coomv_gbyte_count(I M, I N, int64_t nnz, bool beta = false)
{
    return coomv_gbyte_count<T, T, T>(M, N, nnz, beta);
}

template <typename T, typename I, typename J>
constexpr double csrsv_gbyte_count(J M, I nnz)
{
    return ((M + 1) * sizeof(I) + nnz * sizeof(J) + (M + M + nnz) * sizeof(T)) / 1e9;
}

template <typename A, typename X, typename Y, typename I, typename J>
constexpr double csrmv_gbyte_count(J M, J N, I nnz, bool beta = false)
{
    return (sizeof(I) * (M + 1) + sizeof(J) * nnz + sizeof(A) * nnz
            + sizeof(Y) * (M + (beta ? M : 0)) + sizeof(X) * N)
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double csrmv_gbyte_count(J M, J N, I nnz, bool beta = false)
{
    return csrmv_gbyte_count<T, T, T>(M, N, nnz, beta);
}

template <typename T, typename I>
constexpr double gemvi_gbyte_count(I m, I nnz, bool beta = false)
{
    return ((nnz) * sizeof(I) + (m * nnz + nnz + m + (beta ? m : 0)) * sizeof(T)) / 1e9;
}


#endif // GBYTE_HPP