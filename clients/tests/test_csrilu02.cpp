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

#include "testing_csrilu02.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <unistd.h>
#include <vector>
#include <string>

typedef hipsparseIndexBase_t base;
typedef std::tuple<int, base> csrilu02_tuple;
typedef std::tuple<base, std::string> csrilu02_bin_tuple;

int csrilu02_M_range[] = {-1, 0, 50, 647};

base csrilu02_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csrilu02_bin[] = {"mac_econ_fwd500.bin",
                              "mc2depi.bin",
                              "scircuit.bin",
                              "ASIC_320k.bin",
#ifdef __HIP_PLATFORM_HCC__
                              // exclude some matrices from cusparse check,
                              // they use weaker division producing more rounding errors
                              "rma10.bin",
                              "bmwcra_1.bin",
                              "nos1.bin",
                              "nos2.bin",
#endif
                              "nos3.bin",
                              "nos4.bin",
                              "nos5.bin",
                              "nos6.bin",
                              "nos7.bin"};

class parameterized_csrilu02 : public testing::TestWithParam<csrilu02_tuple>
{
    protected:
    parameterized_csrilu02() {}
    virtual ~parameterized_csrilu02() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrilu02_bin : public testing::TestWithParam<csrilu02_bin_tuple>
{
    protected:
    parameterized_csrilu02_bin() {}
    virtual ~parameterized_csrilu02_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrilu02_arguments(csrilu02_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.idx_base = std::get<1>(tup);
    arg.timing   = 0;
    return arg;
}

Arguments setup_csrilu02_arguments(csrilu02_bin_tuple tup)
{
    Arguments arg;
    arg.M        = -99;
    arg.idx_base = std::get<0>(tup);
    arg.timing   = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<1>(tup);

    // Get current executables absolute path
    char path_exe[PATH_MAX];
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
    arg.filename = std::string(path_exe) + "matrices/" + bin_file;

    return arg;
}

TEST(csrilu02_bad_arg, csrilu02_float) { testing_csrilu02_bad_arg<float>(); }

TEST_P(parameterized_csrilu02, csrilu02_float)
{
    Arguments arg = setup_csrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilu02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrilu02, csrilu02_double)
{
    Arguments arg = setup_csrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilu02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrilu02_bin, csrilu02_bin_float)
{
    Arguments arg = setup_csrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilu02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrilu02_bin, csrilu02_bin_double)
{
    Arguments arg = setup_csrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilu02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(csrilu02,
                        parameterized_csrilu02,
                        testing::Combine(testing::ValuesIn(csrilu02_M_range),
                                         testing::ValuesIn(csrilu02_idxbase_range)));

INSTANTIATE_TEST_CASE_P(csrilu02_bin,
                        parameterized_csrilu02_bin,
                        testing::Combine(testing::ValuesIn(csrilu02_idxbase_range),
                                         testing::ValuesIn(csrilu02_bin)));
