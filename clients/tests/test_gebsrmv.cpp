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

#include "testing_gebsrmv.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <hipsparse.h>
#include <string>
#include <unistd.h>
#include <vector>

typedef hipsparseIndexBase_t                                         base;
typedef hipsparseDirection_t                                         dir;
typedef std::tuple<int, int, double, double, int, int, dir, base>    gebsrmv_tuple;
typedef std::tuple<double, double, int, int, dir, base, std::string> gebsrmv_bin_tuple;

int gebsr_M_range[]       = {-1, 0, 500, 7111};
int gebsr_N_range[]       = {-3, 0, 842, 4441};
int gebsr_row_dim_range[] = {5, 9};
int gebsr_col_dim_range[] = {7};

std::vector<double> gebsr_alpha_range = {3.0};
std::vector<double> gebsr_beta_range  = {2.0};

base gebsr_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
dir  gebsr_dir_range[]     = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};

std::string gebsr_bin[] = {"rma10.bin",
                           "mac_econ_fwd500.bin",
                           "bibd_22_8.bin",
                           "mc2depi.bin",
                           "scircuit.bin",
                           "ASIC_320k.bin",
                           "bmwcra_1.bin",
                           "nos1.bin",
                           "nos2.bin",
                           "nos3.bin",
                           "nos4.bin",
                           "nos5.bin",
                           "nos6.bin",
                           "nos7.bin",
                           "amazon0312.bin",
                           "Chebyshev4.bin",
                           "sme3Dc.bin",
                           "webbase-1M.bin",
                           "shipsec1.bin"};

class parameterized_gebsrmv : public testing::TestWithParam<gebsrmv_tuple>
{
protected:
    parameterized_gebsrmv() {}
    virtual ~parameterized_gebsrmv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_gebsrmv_bin : public testing::TestWithParam<gebsrmv_bin_tuple>
{
protected:
    parameterized_gebsrmv_bin() {}
    virtual ~parameterized_gebsrmv_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gebsrmv_arguments(gebsrmv_tuple tup)
{
    Arguments arg;
    arg.M              = std::get<0>(tup);
    arg.N              = std::get<1>(tup);
    arg.alpha          = std::get<2>(tup);
    arg.beta           = std::get<3>(tup);
    arg.row_block_dimA = std::get<4>(tup);
    arg.col_block_dimA = std::get<5>(tup);
    arg.dirA           = std::get<6>(tup);
    arg.idx_base       = std::get<7>(tup);
    arg.timing         = 0;
    return arg;
}

Arguments setup_gebsrmv_arguments(gebsrmv_bin_tuple tup)
{
    Arguments arg;
    arg.M              = -99;
    arg.N              = -99;
    arg.alpha          = std::get<0>(tup);
    arg.beta           = std::get<1>(tup);
    arg.row_block_dimA = std::get<2>(tup);
    arg.col_block_dimA = std::get<3>(tup);
    arg.dirA           = std::get<4>(tup);
    arg.idx_base       = std::get<5>(tup);
    arg.timing         = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<6>(tup);

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

TEST(gebsrmv_bad_arg, gebsrmv_float)
{
    testing_gebsrmv_bad_arg<float>();
}

TEST_P(parameterized_gebsrmv, gebsrmv_float)
{
    Arguments arg = setup_gebsrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsrmv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsrmv, gebsrmv_double)
{
    Arguments arg = setup_gebsrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsrmv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsrmv, gebsrmv_float_complex)
{
    Arguments arg = setup_gebsrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsrmv<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsrmv, gebsrmv_double_complex)
{
    Arguments arg = setup_gebsrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsrmv<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsrmv_bin, gebsrmv_bin_float)
{
    Arguments arg = setup_gebsrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsrmv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsrmv_bin, gebsrmv_bin_double)
{
    Arguments arg = setup_gebsrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsrmv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(gebsrmv,
                        parameterized_gebsrmv,
                        testing::Combine(testing::ValuesIn(gebsr_M_range),
                                         testing::ValuesIn(gebsr_N_range),
                                         testing::ValuesIn(gebsr_alpha_range),
                                         testing::ValuesIn(gebsr_beta_range),
                                         testing::ValuesIn(gebsr_row_dim_range),
                                         testing::ValuesIn(gebsr_col_dim_range),
                                         testing::ValuesIn(gebsr_dir_range),
                                         testing::ValuesIn(gebsr_idxbase_range)));

INSTANTIATE_TEST_CASE_P(gebsrmv_bin,
                        parameterized_gebsrmv_bin,
                        testing::Combine(testing::ValuesIn(gebsr_alpha_range),
                                         testing::ValuesIn(gebsr_beta_range),
                                         testing::ValuesIn(gebsr_row_dim_range),
                                         testing::ValuesIn(gebsr_col_dim_range),
                                         testing::ValuesIn(gebsr_dir_range),
                                         testing::ValuesIn(gebsr_idxbase_range),
                                         testing::ValuesIn(gebsr_bin)));
