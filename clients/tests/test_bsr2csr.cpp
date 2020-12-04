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

#include "testing_bsr2csr.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, int, hipsparseIndexBase_t, hipsparseIndexBase_t, hipsparseDirection_t>
    bsr2csr_tuple;
typedef std::
    tuple<int, hipsparseIndexBase_t, hipsparseIndexBase_t, hipsparseDirection_t, std::string>
        bsr2csr_bin_tuple;

// Random matrices
int bsr2csr_M_range[]         = {-1, 0, 872, 13095, 21453};
int bsr2csr_N_range[]         = {-3, 0, 623, 12766, 29285};
int bsr2csr_block_dim_range[] = {-1, 0, 1, 2, 4, 7, 16};

hipsparseIndexBase_t bsr2csr_csr_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO};

hipsparseIndexBase_t bsr2csr_bsr_base_range[] = {HIPSPARSE_INDEX_BASE_ONE};

hipsparseDirection_t bsr2csr_dir_range[] = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};

// Matrices from files (float and double)
int bsr2csr_block_dim_range_bin[] = {5};

hipsparseIndexBase_t bsr2csr_csr_base_range_bin[] = {HIPSPARSE_INDEX_BASE_ONE};

hipsparseIndexBase_t bsr2csr_bsr_base_range_bin[] = {HIPSPARSE_INDEX_BASE_ONE};

hipsparseDirection_t bsr2csr_dir_range_bin[]
    = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};

std::string bsr2csr_bin[]
    = {"nos1.bin", "nos2.bin", "nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_bsr2csr : public testing::TestWithParam<bsr2csr_tuple>
{
protected:
    parameterized_bsr2csr() {}
    virtual ~parameterized_bsr2csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_bsr2csr_bin : public testing::TestWithParam<bsr2csr_bin_tuple>
{
protected:
    parameterized_bsr2csr_bin() {}
    virtual ~parameterized_bsr2csr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_bsr2csr_arguments(bsr2csr_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.block_dim = std::get<2>(tup);
    arg.idx_base  = std::get<3>(tup);
    arg.idx_base2 = std::get<4>(tup);
    arg.dirA      = std::get<5>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_bsr2csr_arguments(bsr2csr_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.N         = -99;
    arg.block_dim = std::get<0>(tup);
    arg.idx_base  = std::get<1>(tup);
    arg.idx_base2 = std::get<2>(tup);
    arg.dirA      = std::get<3>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<4>(tup);

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

TEST(bsr2csr_bad_arg, bsr2csr)
{
    testing_bsr2csr_bad_arg<float>();
}

TEST_P(parameterized_bsr2csr, bsr2csr_float)
{
    Arguments arg = setup_bsr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_bsr2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsr2csr, bsr2csr_double)
{
    Arguments arg = setup_bsr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_bsr2csr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsr2csr, bsr2csr_float_complex)
{
    Arguments arg = setup_bsr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_bsr2csr<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsr2csr, bsr2csr_double_complex)
{
    Arguments arg = setup_bsr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_bsr2csr<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsr2csr_bin, bsr2csr_bin_float)
{
    Arguments arg = setup_bsr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_bsr2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsr2csr_bin, bsr2csr_bin_double)
{
    Arguments arg = setup_bsr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_bsr2csr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(bsr2csr,
                        parameterized_bsr2csr,
                        testing::Combine(testing::ValuesIn(bsr2csr_M_range),
                                         testing::ValuesIn(bsr2csr_N_range),
                                         testing::ValuesIn(bsr2csr_block_dim_range),
                                         testing::ValuesIn(bsr2csr_bsr_base_range),
                                         testing::ValuesIn(bsr2csr_csr_base_range),
                                         testing::ValuesIn(bsr2csr_dir_range)));

INSTANTIATE_TEST_CASE_P(bsr2csr_bin,
                        parameterized_bsr2csr_bin,
                        testing::Combine(testing::ValuesIn(bsr2csr_block_dim_range_bin),
                                         testing::ValuesIn(bsr2csr_bsr_base_range_bin),
                                         testing::ValuesIn(bsr2csr_csr_base_range_bin),
                                         testing::ValuesIn(bsr2csr_dir_range_bin),
                                         testing::ValuesIn(bsr2csr_bin)));
