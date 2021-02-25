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

#include "testing_prune_dense2csr.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <hipsparse.h>
#include <string>
#include <vector>

typedef hipsparseIndexBase_t                    base;
typedef std::tuple<int, int, int, double, base> prune_dense2csr_tuple;
int    prune_dense2csr_M_range[]         = {-1, 0, 10, 500, 872, 1000};
int    prune_dense2csr_N_range[]         = {-3, 0, 33, 242, 623, 1000};
int    prune_dense2csr_LD_range[]        = {50, 500, 1000};
double prune_dense2csr_threshold_range[] = {0.1, 0.55};
base   prune_dense2csr_idx_base_range[]  = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_prune_dense2csr : public testing::TestWithParam<prune_dense2csr_tuple>
{
protected:
    parameterized_prune_dense2csr() {}
    virtual ~parameterized_prune_dense2csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_prune_dense2csr_arguments(prune_dense2csr_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.lda       = std::get<2>(tup);
    arg.threshold = std::get<3>(tup);
    arg.idx_base  = std::get<4>(tup);
    return arg;
}

// Only run tests for CUDA 11.1 or greater
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11010)
TEST(prune_dense2csr_bad_arg, prune_dense2csr)
{
    testing_prune_dense2csr_bad_arg<float>();
}

TEST_P(parameterized_prune_dense2csr, prune_dense2csr_float)
{
    Arguments arg = setup_prune_dense2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_prune_dense2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_prune_dense2csr, prune_dense2csr_double)
{
    Arguments arg = setup_prune_dense2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_prune_dense2csr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
#endif

INSTANTIATE_TEST_CASE_P(prune_dense2csr,
                        parameterized_prune_dense2csr,
                        testing::Combine(testing::ValuesIn(prune_dense2csr_M_range),
                                         testing::ValuesIn(prune_dense2csr_N_range),
                                         testing::ValuesIn(prune_dense2csr_LD_range),
                                         testing::ValuesIn(prune_dense2csr_threshold_range),
                                         testing::ValuesIn(prune_dense2csr_idx_base_range)));
