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

#include "testing_spvv.hpp"

#include <hipsparse.h>

typedef std::tuple<int, int, hipsparseIndexBase_t>    spvv_tuple;
typedef std::tuple<hipsparseIndexBase_t, std::string> spvv_bin_tuple;

int spvv_N_range[]   = {50, 750, 2135};
int spvv_nnz_range[] = {5, 45};

hipsparseIndexBase_t spvv_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_spvv : public testing::TestWithParam<spvv_tuple>
{
protected:
    parameterized_spvv() {}
    virtual ~parameterized_spvv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_spvv_arguments(spvv_tuple tup)
{
    Arguments arg;
    arg.N        = std::get<0>(tup);
    arg.nnz      = std::get<1>(tup);
    arg.idx_base = std::get<2>(tup);
    arg.timing   = 0;
    return arg;
}

// csr format not supported in cusparse
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11010)
TEST(spvv_bad_arg, spvv_float)
{
    testing_spvv_bad_arg();
}

TEST_P(parameterized_spvv, spvv_i32_float)
{
    Arguments arg = setup_spvv_arguments(GetParam());

    hipsparseStatus_t status = testing_spvv<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spvv, spvv_i64_double)
{
    Arguments arg = setup_spvv_arguments(GetParam());

    hipsparseStatus_t status = testing_spvv<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spvv, spvv_i32_float_complex)
{
    Arguments arg = setup_spvv_arguments(GetParam());

    hipsparseStatus_t status = testing_spvv<int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spvv, spvv_i64_double_complex)
{
    Arguments arg = setup_spvv_arguments(GetParam());

    hipsparseStatus_t status = testing_spvv<int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(spvv,
                         parameterized_spvv,
                         testing::Combine(testing::ValuesIn(spvv_N_range),
                                          testing::ValuesIn(spvv_nnz_range),
                                          testing::ValuesIn(spvv_idxbase_range)));
#endif
