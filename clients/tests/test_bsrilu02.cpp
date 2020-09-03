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

#include "testing_bsrilu02.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <hipsparse.h>
#include <string>
#include <unistd.h>

typedef hipsparseIndexBase_t                    base;
typedef hipsparseDirection_t                    dir;
typedef std::tuple<int, int, dir, base>         bsrilu02_tuple;
typedef std::tuple<int, dir, base, std::string> bsrilu02_bin_tuple;

int bsrilu02_M_range[]   = {-1, 0, 50, 426};
int bsrilu02_dim_range[] = {-1, 0, 1, 3, 5, 9};

base bsrilu02_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
dir  bsrilu02_dir_range[]     = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};

std::string bsrilu02_bin[] = {"mac_econ_fwd500.bin",
                              "nos1.bin",
                              "nos2.bin",
                              "nos3.bin",
                              "nos4.bin",
                              "nos5.bin",
                              "nos6.bin",
                              "nos7.bin",
                              "scircuit.bin",
                              "mc2depi.bin"};

class parameterized_bsrilu02 : public testing::TestWithParam<bsrilu02_tuple>
{
protected:
    parameterized_bsrilu02() {}
    virtual ~parameterized_bsrilu02() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_bsrilu02_bin : public testing::TestWithParam<bsrilu02_bin_tuple>
{
protected:
    parameterized_bsrilu02_bin() {}
    virtual ~parameterized_bsrilu02_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_bsrilu02_arguments(bsrilu02_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.block_dim = std::get<1>(tup);
    arg.dirA      = std::get<2>(tup);
    arg.idx_base  = std::get<3>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_bsrilu02_arguments(bsrilu02_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.block_dim = std::get<0>(tup);
    arg.dirA      = std::get<1>(tup);
    arg.idx_base  = std::get<2>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<3>(tup);

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

TEST(bsrilu02_bad_arg, bsrilu02_float)
{
    testing_bsrilu02_bad_arg<float>();
}

TEST_P(parameterized_bsrilu02, bsrilu02_float)
{
    Arguments arg = setup_bsrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrilu02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrilu02, bsrilu02_double)
{
    Arguments arg = setup_bsrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrilu02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrilu02, bsrilu02_float_complex)
{
    Arguments arg = setup_bsrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrilu02<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrilu02, bsrilu02_double_complex)
{
    Arguments arg = setup_bsrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrilu02<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrilu02_bin, bsrilu02_bin_float)
{
    Arguments arg = setup_bsrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrilu02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrilu02_bin, bsrilu02_bin_double)
{
    Arguments arg = setup_bsrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrilu02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(bsrilu02,
                        parameterized_bsrilu02,
                        testing::Combine(testing::ValuesIn(bsrilu02_M_range),
                                         testing::ValuesIn(bsrilu02_dim_range),
                                         testing::ValuesIn(bsrilu02_dir_range),
                                         testing::ValuesIn(bsrilu02_idxbase_range)));

INSTANTIATE_TEST_CASE_P(bsrilu02_bin,
                        parameterized_bsrilu02_bin,
                        testing::Combine(testing::ValuesIn(bsrilu02_dim_range),
                                         testing::ValuesIn(bsrilu02_dir_range),
                                         testing::ValuesIn(bsrilu02_idxbase_range),
                                         testing::ValuesIn(bsrilu02_bin)));
