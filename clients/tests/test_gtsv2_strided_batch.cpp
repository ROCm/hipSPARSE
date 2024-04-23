/* ************************************************************************
 * Copyright (C) 2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_gtsv2_strided_batch.hpp"
#include "hipsparse_arguments.hpp"

#include <hipsparse.h>

// Only run tests for CUDA 11.1 or greater
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11010)
TEST(gtsv2_strided_batch_bad_arg, gtsv2_strided_batch_float)
{
    testing_gtsv2_strided_batch_bad_arg<float>();
}

TEST(gtsv2_strided_batch, gtsv2_strided_batch_float)
{
    Arguments arg;
    hipsparseStatus_t status = testing_gtsv2_strided_batch<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(gtsv2_strided_batch, gtsv2_strided_batch_double)
{
    Arguments arg;
    hipsparseStatus_t status = testing_gtsv2_strided_batch<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(gtsv2_strided_batch, gtsv2_strided_batch_hipComplex)
{
    Arguments arg;
    hipsparseStatus_t status = testing_gtsv2_strided_batch<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
#endif
