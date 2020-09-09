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

#include "testing_prune_csr2csr_by_percentage.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, double, hipsparseIndexBase_t, hipsparseIndexBase_t>
    prune_csr2csr_by_percentage_tuple;
typedef std::tuple<double, hipsparseIndexBase_t, hipsparseIndexBase_t, std::string>
    prune_csr2csr_by_percentage_bin_tuple;

int    prune_csr2csr_by_percentage_M_range[] = {-1, 0, 10, 500, 872, 465327};
int    prune_csr2csr_by_percentage_N_range[] = {-3, 0, 33, 242, 623, 592645};
double prune_csr2csr_by_percentage_range[]   = {5.7, 75.0, 90.0, 101.0};

hipsparseIndexBase_t prune_csr2csr_by_percentage_base_A_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

hipsparseIndexBase_t prune_csr2csr_by_percentage_base_C_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string prune_csr2csr_by_percentage_bin[] = {"rma10.bin",
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

class parameterized_prune_csr2csr_by_percentage
    : public testing::TestWithParam<prune_csr2csr_by_percentage_tuple>
{
protected:
    parameterized_prune_csr2csr_by_percentage() {}
    virtual ~parameterized_prune_csr2csr_by_percentage() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_prune_csr2csr_by_percentage_bin
    : public testing::TestWithParam<prune_csr2csr_by_percentage_bin_tuple>
{
protected:
    parameterized_prune_csr2csr_by_percentage_bin() {}
    virtual ~parameterized_prune_csr2csr_by_percentage_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_prune_csr2csr_by_percentage_arguments(prune_csr2csr_by_percentage_tuple tup)
{
    Arguments arg;
    arg.M          = std::get<0>(tup);
    arg.N          = std::get<1>(tup);
    arg.percentage = std::get<2>(tup);
    arg.idx_base   = std::get<3>(tup);
    arg.idx_base2  = std::get<4>(tup);
    arg.timing     = 0;
    return arg;
}

Arguments setup_prune_csr2csr_by_percentage_arguments(prune_csr2csr_by_percentage_bin_tuple tup)
{
    Arguments arg;
    arg.M          = -99;
    arg.N          = -99;
    arg.percentage = std::get<0>(tup);
    arg.idx_base   = std::get<1>(tup);
    arg.idx_base2  = std::get<2>(tup);
    arg.timing     = 0;

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

TEST(prune_csr2csr_by_percentage_bad_arg, prune_csr2csr_by_percentage)
{
    testing_prune_csr2csr_by_percentage_bad_arg<float>();
}

TEST_P(parameterized_prune_csr2csr_by_percentage, prune_csr2csr_by_percentage_float)
{
    Arguments arg = setup_prune_csr2csr_by_percentage_arguments(GetParam());

    hipsparseStatus_t status = testing_prune_csr2csr_by_percentage<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_prune_csr2csr_by_percentage, prune_csr2csr_by_percentage_double)
{
    Arguments arg = setup_prune_csr2csr_by_percentage_arguments(GetParam());

    hipsparseStatus_t status = testing_prune_csr2csr_by_percentage<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_prune_csr2csr_by_percentage_bin, prune_csr2csr_by_percentage_bin_float)
{
    Arguments arg = setup_prune_csr2csr_by_percentage_arguments(GetParam());

    hipsparseStatus_t status = testing_prune_csr2csr_by_percentage<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_prune_csr2csr_by_percentage_bin, prune_csr2csr_percentage_bin_double)
{
    Arguments arg = setup_prune_csr2csr_by_percentage_arguments(GetParam());

    hipsparseStatus_t status = testing_prune_csr2csr_by_percentage<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(
    prune_csr2csr_by_percentage,
    parameterized_prune_csr2csr_by_percentage,
    testing::Combine(testing::ValuesIn(prune_csr2csr_by_percentage_M_range),
                     testing::ValuesIn(prune_csr2csr_by_percentage_N_range),
                     testing::ValuesIn(prune_csr2csr_by_percentage_range),
                     testing::ValuesIn(prune_csr2csr_by_percentage_base_A_range),
                     testing::ValuesIn(prune_csr2csr_by_percentage_base_C_range)));

INSTANTIATE_TEST_CASE_P(
    prune_csr2csr_by_percentage_bin,
    parameterized_prune_csr2csr_by_percentage_bin,
    testing::Combine(testing::ValuesIn(prune_csr2csr_by_percentage_range),
                     testing::ValuesIn(prune_csr2csr_by_percentage_base_A_range),
                     testing::ValuesIn(prune_csr2csr_by_percentage_base_C_range),
                     testing::ValuesIn(prune_csr2csr_by_percentage_bin)));
