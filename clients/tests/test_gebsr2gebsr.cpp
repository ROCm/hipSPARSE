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

#include "testing_gebsr2gebsr.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int,
                   int,
                   int,
                   int,
                   int,
                   int,
                   hipsparseIndexBase_t,
                   hipsparseIndexBase_t,
                   hipsparseDirection_t>
    gebsr2gebsr_tuple;
typedef std::tuple<int,
                   int,
                   int,
                   int,
                   hipsparseIndexBase_t,
                   hipsparseIndexBase_t,
                   hipsparseDirection_t,
                   std::string>
    gebsr2gebsr_bin_tuple;

// Random matrices
int gebsr2gebsr_M_range[]               = {-1, 0, 872, 13095, 21453};
int gebsr2gebsr_N_range[]               = {-3, 0, 623, 12766, 29285};
int gebsr2gebsr_row_block_dim_A_range[] = {-1, 0, 2};
int gebsr2gebsr_col_block_dim_A_range[] = {-1, 0, 5};
int gebsr2gebsr_row_block_dim_C_range[] = {-1, 0, 3};
int gebsr2gebsr_col_block_dim_C_range[] = {-1, 0, 4};

hipsparseIndexBase_t gebsr2gebsr_A_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO};
hipsparseIndexBase_t gebsr2gebsr_C_base_range[] = {HIPSPARSE_INDEX_BASE_ONE};
hipsparseDirection_t gebsr2gebsr_dir_range[]
    = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};

// Matrices from files (float and double)
int gebsr2gebsr_row_block_dim_A_range_bin[] = {5};
int gebsr2gebsr_col_block_dim_A_range_bin[] = {4};
int gebsr2gebsr_row_block_dim_C_range_bin[] = {2};
int gebsr2gebsr_col_block_dim_C_range_bin[] = {7};

hipsparseIndexBase_t gebsr2gebsr_A_base_range_bin[] = {HIPSPARSE_INDEX_BASE_ONE};
hipsparseIndexBase_t gebsr2gebsr_C_base_range_bin[] = {HIPSPARSE_INDEX_BASE_ONE};

hipsparseDirection_t gebsr2gebsr_dir_range_bin[]
    = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};

std::string gebsr2gebsr_bin[] = {"rma10.bin",
                                 "mac_econ_fwd500.bin",
                                 "nos1.bin",
                                 "nos2.bin",
                                 "nos3.bin",
                                 "nos4.bin",
                                 "nos5.bin",
                                 "nos6.bin",
                                 "nos7.bin"};

class parameterized_gebsr2gebsr : public testing::TestWithParam<gebsr2gebsr_tuple>
{
protected:
    parameterized_gebsr2gebsr() {}
    virtual ~parameterized_gebsr2gebsr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_gebsr2gebsr_bin : public testing::TestWithParam<gebsr2gebsr_bin_tuple>
{
protected:
    parameterized_gebsr2gebsr_bin() {}
    virtual ~parameterized_gebsr2gebsr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gebsr2gebsr_arguments(gebsr2gebsr_tuple tup)
{
    Arguments arg;
    arg.M              = std::get<0>(tup);
    arg.N              = std::get<1>(tup);
    arg.row_block_dimA = std::get<2>(tup);
    arg.col_block_dimA = std::get<3>(tup);
    arg.row_block_dimB = std::get<4>(tup);
    arg.col_block_dimB = std::get<5>(tup);
    arg.idx_base       = std::get<6>(tup);
    arg.idx_base2      = std::get<7>(tup);
    arg.dirA           = std::get<8>(tup);
    arg.timing         = 0;
    return arg;
}

Arguments setup_gebsr2gebsr_arguments(gebsr2gebsr_bin_tuple tup)
{
    Arguments arg;
    arg.M              = -99;
    arg.N              = -99;
    arg.row_block_dimA = std::get<0>(tup);
    arg.col_block_dimA = std::get<1>(tup);
    arg.row_block_dimB = std::get<2>(tup);
    arg.col_block_dimB = std::get<3>(tup);
    arg.idx_base       = std::get<4>(tup);
    arg.idx_base2      = std::get<5>(tup);
    arg.dirA           = std::get<6>(tup);
    arg.timing         = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<7>(tup);

    // Get current executables absolute path
    char    path_exe[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", path_exe, sizeof(path_exe) - 1);
    if(len < 14)
    {
        path_exe[0] = '\0';
    }
    else
    {
        path_exe[len - 14] = '\0';
    }

    // Matrices are stored at the same path in matrices directory
    arg.filename = std::string(path_exe) + "../matrices/" + bin_file;

    return arg;
}

TEST(gebsr2gebsr_bad_arg, gebsr2gebsr)
{
    testing_gebsr2gebsr_bad_arg<float>();
}

TEST_P(parameterized_gebsr2gebsr, gebsr2gebsr_float)
{
    Arguments arg = setup_gebsr2gebsr_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2gebsr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2gebsr, gebsr2gebsr_double)
{
    Arguments arg = setup_gebsr2gebsr_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2gebsr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2gebsr, gebsr2gebsr_float_complex)
{
    Arguments arg = setup_gebsr2gebsr_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2gebsr<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2gebsr, gebsr2gebsr_double_complex)
{
    Arguments arg = setup_gebsr2gebsr_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2gebsr<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2gebsr_bin, gebsr2gebsr_bin_float)
{
    Arguments arg = setup_gebsr2gebsr_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2gebsr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2gebsr_bin, gebsr2gebsr_bin_double)
{
    Arguments arg = setup_gebsr2gebsr_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2gebsr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(gebsr2gebsr,
                        parameterized_gebsr2gebsr,
                        testing::Combine(testing::ValuesIn(gebsr2gebsr_M_range),
                                         testing::ValuesIn(gebsr2gebsr_N_range),
                                         testing::ValuesIn(gebsr2gebsr_row_block_dim_A_range),
                                         testing::ValuesIn(gebsr2gebsr_col_block_dim_A_range),
                                         testing::ValuesIn(gebsr2gebsr_row_block_dim_C_range),
                                         testing::ValuesIn(gebsr2gebsr_col_block_dim_C_range),
                                         testing::ValuesIn(gebsr2gebsr_A_base_range),
                                         testing::ValuesIn(gebsr2gebsr_C_base_range),
                                         testing::ValuesIn(gebsr2gebsr_dir_range)));

INSTANTIATE_TEST_CASE_P(gebsr2gebsr_bin,
                        parameterized_gebsr2gebsr_bin,
                        testing::Combine(testing::ValuesIn(gebsr2gebsr_row_block_dim_A_range_bin),
                                         testing::ValuesIn(gebsr2gebsr_col_block_dim_A_range_bin),
                                         testing::ValuesIn(gebsr2gebsr_row_block_dim_C_range_bin),
                                         testing::ValuesIn(gebsr2gebsr_col_block_dim_C_range_bin),
                                         testing::ValuesIn(gebsr2gebsr_A_base_range_bin),
                                         testing::ValuesIn(gebsr2gebsr_C_base_range_bin),
                                         testing::ValuesIn(gebsr2gebsr_dir_range_bin),
                                         testing::ValuesIn(gebsr2gebsr_bin)));
