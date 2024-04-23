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

#include "testing_bsrsm2.hpp"
#include "hipsparse_arguments.hpp"

#include <hipsparse.h>

// Only run tests for CUDA 11.1 or greater
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11010)
TEST(bsrsm2_bad_arg, bsrsm2_float)
{
    testing_bsrsm2_bad_arg();
}

TEST(bsrsm2, bsrsm2_float)
{
    Arguments arg;
    hipsparseStatus_t status = testing_bsrsm2<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(bsrsm2, bsrsm2_double)
{
    Arguments arg;
    hipsparseStatus_t status = testing_bsrsm2<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(bsrsm2, bsrsm2_hipComplex)
{
    Arguments arg;
    hipsparseStatus_t status = testing_bsrsm2<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(bsrsm2, bsrsm2_hipDoubleComplex)
{
    Arguments arg;
    hipsparseStatus_t status = testing_bsrsm2<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
#endif
