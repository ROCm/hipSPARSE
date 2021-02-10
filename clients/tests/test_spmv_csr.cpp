/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include "testing_spmv_csr.hpp"

#include <gtest/gtest.h>
#include <hipsparse.h>

TEST(spmv_csr_bad_arg, spmv_csr_float)
{
    testing_spmv_csr_bad_arg();
}

TEST(spmv_csr, spmv_csr_i32_i32_float)
{
    hipsparseStatus_t status = testing_spmv_csr<int32_t, int32_t, float>();
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

#if(!defined(CUDART_VERSION))
TEST(spmv_csr, spmv_csr_i64_i32_double)
{
    hipsparseStatus_t status = testing_spmv_csr<int64_t, int32_t, double>();
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
#endif

TEST(spmv_csr, spmv_csr_i64_i64_hipComplex)
{
    hipsparseStatus_t status = testing_spmv_csr<int64_t, int64_t, hipComplex>();
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
