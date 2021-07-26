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

#include "testing_csrcolor.hpp"


#include <hipsparse.h>

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11010)

TEST(csrcolor_bad_arg, csrcolor_bad_arg_float)
{
    testing_csrcolor_bad_arg<float>();
}

TEST(csrcolor, csrcolor_float)
{
    hipsparseStatus_t status = testing_csrcolor<float>();
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(csrcolor, csrcolor_double)
{
    hipsparseStatus_t status = testing_csrcolor<double>();
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(csrcolor, csrcolor_hipComplex)
{
    hipsparseStatus_t status = testing_csrcolor<hipComplex>();
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(csrcolor, csrcolor_hipDoubleComplex)
{
    hipsparseStatus_t status = testing_csrcolor<hipDoubleComplex>();
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

#endif
