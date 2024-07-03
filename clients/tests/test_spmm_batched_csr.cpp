/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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


#include "testing_spmm_batched_csr.hpp"

#include <hipsparse.h>

// Only run tests for CUDA 11.1 or greater
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11010)
TEST(spmm_batched_csr_bad_arg, spmm_batched_csr_float)
{
    testing_spmm_batched_csr_bad_arg();
}

TEST(spmm_batched_csr, spmm_batched_csr_i32_i32_float)
{
    hipsparseStatus_t status = testing_spmm_batched_csr<int32_t, int32_t, float>();
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(spmm_batched_csr, spmm_batched_csr_i32_i32_double)
{
    hipsparseStatus_t status = testing_spmm_batched_csr<int32_t, int32_t, double>();
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

// TEST(spmm_batched_csr, spmm_batched_csr_i32_i32_hipComplex)
// {
//     hipsparseStatus_t status = testing_spmm_batched_csr<int32_t, int32_t, hipComplex>();
//     EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
// }
#endif
