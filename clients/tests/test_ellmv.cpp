/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_ellmv.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>

typedef hipsparseIndexBase_t base;
typedef std::tuple<int, int, double, double, base> ellmv_tuple;
typedef std::tuple<double, double, base, std::string> ellmv_bin_tuple;

int ell_M_range[] = {-1, 0, 10, 500, 7111, 10000};
int ell_N_range[] = {-3, 0, 33, 842, 4441, 10000};

std::vector<double> ell_alpha_range = {2.0, 3.0};
std::vector<double> ell_beta_range  = {0.0, 0.6};

base ell_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string ell_bin[] = {"rma10.bin",
                         "mac_econ_fwd500.bin",
                         "bibd_22_8.bin",
                         "mc2depi.bin",
                         "scircuit.bin",
                         "bmwcra_1.bin",
                         "nos1.bin",
                         "nos2.bin",
                         "nos3.bin",
                         "nos4.bin",
                         "nos5.bin",
                         "nos6.bin",
                         "nos7.bin"};

class parameterized_ellmv : public testing::TestWithParam<ellmv_tuple>
{
    protected:
    parameterized_ellmv() {}
    virtual ~parameterized_ellmv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_ellmv_bin : public testing::TestWithParam<ellmv_bin_tuple>
{
    protected:
    parameterized_ellmv_bin() {}
    virtual ~parameterized_ellmv_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_ellmv_arguments(ellmv_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.alpha    = std::get<2>(tup);
    arg.beta     = std::get<3>(tup);
    arg.idx_base = std::get<4>(tup);
    arg.timing   = 0;
    return arg;
}

Arguments setup_ellmv_arguments(ellmv_bin_tuple tup)
{
    Arguments arg;
    arg.M        = -99;
    arg.N        = -99;
    arg.alpha    = std::get<0>(tup);
    arg.beta     = std::get<1>(tup);
    arg.idx_base = std::get<2>(tup);
    arg.timing   = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<3>(tup);

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

TEST(ellmv_bad_arg, ellmv_float) { testing_ellmv_bad_arg<float>(); }

TEST_P(parameterized_ellmv, ellmv_float)
{
    Arguments arg = setup_ellmv_arguments(GetParam());

    hipsparseStatus_t status = testing_ellmv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_ellmv, ellmv_double)
{
    Arguments arg = setup_ellmv_arguments(GetParam());

    hipsparseStatus_t status = testing_ellmv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_ellmv_bin, ellmv_bin_float)
{
    Arguments arg = setup_ellmv_arguments(GetParam());

    hipsparseStatus_t status = testing_ellmv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_ellmv_bin, ellmv_bin_double)
{
    Arguments arg = setup_ellmv_arguments(GetParam());

    hipsparseStatus_t status = testing_ellmv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(ellmv,
                        parameterized_ellmv,
                        testing::Combine(testing::ValuesIn(ell_M_range),
                                         testing::ValuesIn(ell_N_range),
                                         testing::ValuesIn(ell_alpha_range),
                                         testing::ValuesIn(ell_beta_range),
                                         testing::ValuesIn(ell_idxbase_range)));

INSTANTIATE_TEST_CASE_P(ellmv_bin,
                        parameterized_ellmv_bin,
                        testing::Combine(testing::ValuesIn(ell_alpha_range),
                                         testing::ValuesIn(ell_beta_range),
                                         testing::ValuesIn(ell_idxbase_range),
                                         testing::ValuesIn(ell_bin)));
