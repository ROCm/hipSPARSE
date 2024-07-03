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


#include "testing_sddmm_coo.hpp"

#include <hipsparse.h>

typedef std::tuple<int,
                   int,
                   int,
                   double,
                   double,
                   hipsparseOperation_t,
                   hipsparseOperation_t,
                   hipsparseOrder_t,
                   hipsparseIndexBase_t>
    sddmm_coo_tuple;
typedef std::tuple<int,
                   double,
                   double,
                   hipsparseOperation_t,
                   hipsparseOperation_t,
                   hipsparseOrder_t,
                   hipsparseIndexBase_t,
                   std::string>
    sddmm_coo_bin_tuple;

int sddmm_coo_M_range[] = {50};
int sddmm_coo_N_range[] = {84};
int sddmm_coo_K_range[] = {5};

std::vector<double> sddmm_coo_alpha_range = {2.0};
std::vector<double> sddmm_coo_beta_range  = {1.0};

hipsparseOperation_t sddmm_coo_transA_range[] = {HIPSPARSE_OPERATION_NON_TRANSPOSE};
hipsparseOperation_t sddmm_coo_transB_range[] = {HIPSPARSE_OPERATION_NON_TRANSPOSE};
hipsparseOrder_t     sddmm_coo_order_range[]  = {HIPSPARSE_ORDER_COL};
hipsparseIndexBase_t sddmm_coo_idxbase_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string sddmm_coo_bin[] = {"nos1.bin",
                               "nos2.bin",
                               "nos3.bin",
                               "nos4.bin",
                               "nos5.bin",
                               "nos6.bin",
                               "nos7.bin",
                               "Chebyshev4.bin",
                               "shipsec1.bin"};

class parameterized_sddmm_coo : public testing::TestWithParam<sddmm_coo_tuple>
{
protected:
    parameterized_sddmm_coo() {}
    virtual ~parameterized_sddmm_coo() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_sddmm_coo_bin : public testing::TestWithParam<sddmm_coo_bin_tuple>
{
protected:
    parameterized_sddmm_coo_bin() {}
    virtual ~parameterized_sddmm_coo_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_sddmm_coo_arguments(sddmm_coo_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.K        = std::get<2>(tup);
    arg.alpha    = std::get<3>(tup);
    arg.beta     = std::get<4>(tup);
    arg.transA   = std::get<5>(tup);
    arg.transB   = std::get<6>(tup);
    arg.orderA   = std::get<7>(tup);
    arg.idx_base = std::get<8>(tup);
    arg.timing   = 0;
    return arg;
}

Arguments setup_sddmm_coo_arguments(sddmm_coo_bin_tuple tup)
{
    Arguments arg;
    arg.M        = -99;
    arg.N        = -99;
    arg.K        = std::get<0>(tup);
    arg.alpha    = std::get<1>(tup);
    arg.beta     = std::get<2>(tup);
    arg.transA   = std::get<3>(tup);
    arg.transB   = std::get<4>(tup);
    arg.orderA   = std::get<5>(tup);
    arg.idx_base = std::get<6>(tup);
    arg.timing   = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<7>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.filename = get_filename(bin_file);

    return arg;
}

// COO format not supported in cusparse
#if(!defined(CUDART_VERSION))
TEST(sddmm_coo_bad_arg, sddmm_coo_float)
{
    testing_sddmm_coo_bad_arg();
}

TEST_P(parameterized_sddmm_coo, sddmm_coo_i32_float)
{
    Arguments arg = setup_sddmm_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_sddmm_coo<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sddmm_coo, sddmm_coo_i64_double)
{
    Arguments arg = setup_sddmm_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_sddmm_coo<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sddmm_coo, sddmm_coo_i32_float_complex)
{
    Arguments arg = setup_sddmm_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_sddmm_coo<int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sddmm_coo, sddmm_coo_i64_double_complex)
{
    Arguments arg = setup_sddmm_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_sddmm_coo<int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sddmm_coo_bin, sddmm_coo_bin_i32_float)
{
    Arguments arg = setup_sddmm_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_sddmm_coo<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sddmm_coo_bin, sddmm_coo_bin_i64_double)
{
    Arguments arg = setup_sddmm_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_sddmm_coo<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(sddmm_coo,
                         parameterized_sddmm_coo,
                         testing::Combine(testing::ValuesIn(sddmm_coo_M_range),
                                          testing::ValuesIn(sddmm_coo_N_range),
                                          testing::ValuesIn(sddmm_coo_K_range),
                                          testing::ValuesIn(sddmm_coo_alpha_range),
                                          testing::ValuesIn(sddmm_coo_beta_range),
                                          testing::ValuesIn(sddmm_coo_transA_range),
                                          testing::ValuesIn(sddmm_coo_transB_range),
                                          testing::ValuesIn(sddmm_coo_order_range),
                                          testing::ValuesIn(sddmm_coo_idxbase_range)));

INSTANTIATE_TEST_SUITE_P(sddmm_coo_bin,
                         parameterized_sddmm_coo_bin,
                         testing::Combine(testing::ValuesIn(sddmm_coo_K_range),
                                          testing::ValuesIn(sddmm_coo_alpha_range),
                                          testing::ValuesIn(sddmm_coo_beta_range),
                                          testing::ValuesIn(sddmm_coo_transA_range),
                                          testing::ValuesIn(sddmm_coo_transB_range),
                                          testing::ValuesIn(sddmm_coo_order_range),
                                          testing::ValuesIn(sddmm_coo_idxbase_range),
                                          testing::ValuesIn(sddmm_coo_bin)));
#endif
