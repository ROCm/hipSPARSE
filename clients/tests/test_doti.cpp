/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

#include "testing_doti.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <hipsparse.h>
#include <vector>

typedef hipsparseIndexBase_t       base;
typedef std::tuple<int, int, base> doti_tuple;

int doti_N_range[]   = {12000, 15332, 22031};
int doti_nnz_range[] = {-1, 0, 5, 10, 500, 1000, 7111, 10000};

base doti_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_doti : public testing::TestWithParam<doti_tuple>
{
protected:
    parameterized_doti() {}
    virtual ~parameterized_doti() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_doti_arguments(doti_tuple tup)
{
    Arguments arg;
    arg.N        = std::get<0>(tup);
    arg.nnz      = std::get<1>(tup);
    arg.idx_base = std::get<2>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(doti_bad_arg, doti_float)
{
    testing_doti_bad_arg<float>();
}

TEST_P(parameterized_doti, doti_float)
{
    Arguments arg = setup_doti_arguments(GetParam());

    hipsparseStatus_t status = testing_doti<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_doti, doti_double)
{
    Arguments arg = setup_doti_arguments(GetParam());

    hipsparseStatus_t status = testing_doti<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(doti,
                        parameterized_doti,
                        testing::Combine(testing::ValuesIn(doti_N_range),
                                         testing::ValuesIn(doti_nnz_range),
                                         testing::ValuesIn(doti_idx_base_range)));
