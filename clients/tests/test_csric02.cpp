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

#include "testing_csric02.hpp"
#include "utility.hpp"

#include <hipsparse/hipsparse.h>
#include <string>
#include <vector>

typedef hipsparseIndexBase_t          base;
typedef std::tuple<int, base>         csric02_tuple;
typedef std::tuple<base, std::string> csric02_bin_tuple;

int csric02_M_range[] = {-1, 0, 50, 426};

base csric02_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csric02_bin[] = {"nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_csric02 : public testing::TestWithParam<csric02_tuple>
{
protected:
    parameterized_csric02() {}
    virtual ~parameterized_csric02() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csric02_bin : public testing::TestWithParam<csric02_bin_tuple>
{
protected:
    parameterized_csric02_bin() {}
    virtual ~parameterized_csric02_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csric02_arguments(csric02_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.idx_base = std::get<1>(tup);
    arg.timing   = 0;
    return arg;
}

Arguments setup_csric02_arguments(csric02_bin_tuple tup)
{
    Arguments arg;
    arg.M        = -99;
    arg.idx_base = std::get<0>(tup);
    arg.timing   = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<1>(tup);

    // Get current executables absolute path

    // Matrices are stored at the same path in matrices directory
    arg.filename = hipsparse_exepath() + "../matrices/" + bin_file;

    return arg;
}

// Only run tests for CUDA 11.1 or greater
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11010)
TEST(csric02_bad_arg, csric02_float)
{
    testing_csric02_bad_arg<float>();
}

TEST_P(parameterized_csric02, csric02_float)
{
    Arguments arg = setup_csric02_arguments(GetParam());

    hipsparseStatus_t status = testing_csric02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csric02, csric02_double)
{
    Arguments arg = setup_csric02_arguments(GetParam());

    hipsparseStatus_t status = testing_csric02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csric02, csric02_float_complex)
{
    Arguments arg = setup_csric02_arguments(GetParam());

    hipsparseStatus_t status = testing_csric02<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csric02, csric02_double_complex)
{
    Arguments arg = setup_csric02_arguments(GetParam());

    hipsparseStatus_t status = testing_csric02<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csric02_bin, csric02_bin_float)
{
    Arguments arg = setup_csric02_arguments(GetParam());

    hipsparseStatus_t status = testing_csric02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csric02_bin, csric02_bin_double)
{
    Arguments arg = setup_csric02_arguments(GetParam());

    hipsparseStatus_t status = testing_csric02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
#endif

INSTANTIATE_TEST_SUITE_P(csric02,
                         parameterized_csric02,
                         testing::Combine(testing::ValuesIn(csric02_M_range),
                                          testing::ValuesIn(csric02_idxbase_range)));

INSTANTIATE_TEST_SUITE_P(csric02_bin,
                         parameterized_csric02_bin,
                         testing::Combine(testing::ValuesIn(csric02_idxbase_range),
                                          testing::ValuesIn(csric02_bin)));
