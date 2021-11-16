/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include "testing_bsrxmv.hpp"

#include <hipsparse/hipsparse.h>

TEST(bsrxmv_bad_arg, bsrxmv_bad_arg_float)
{
    testing_bsrxmv_bad_arg<float>();
}

TEST(bsrxmv_bad_arg, bsrxmv_bad_arg_double)
{
    testing_bsrxmv_bad_arg<double>();
}

TEST(bsrxmv_bad_arg, bsrxmv_bad_arg_float_complex)
{
    testing_bsrxmv_bad_arg<hipComplex>();
}

TEST(bsrxmv_bad_arg, bsrxmv_bad_arg_double_complex)
{
    testing_bsrxmv_bad_arg<hipDoubleComplex>();
}

TEST(bsrxmv, bsrxmv_float)
{
    hipsparseStatus_t status = testing_bsrxmv<float>();
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(bsrxmv, bsrxmv_double)
{
    hipsparseStatus_t status = testing_bsrxmv<double>();
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(bsrxmv, bsrxmv_hipComplex)
{
    hipsparseStatus_t status = testing_bsrxmv<hipComplex>();
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
