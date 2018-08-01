/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrmv.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <unistd.h>
#include <vector>
#include <string>

typedef hipsparseIndexBase_t base;
typedef std::tuple<int, int, double, double, base> csrmv_tuple;
typedef std::tuple<double, double, base, std::string> csrmv_bin_tuple;

int csr_M_range[] = {-1, 0, 500, 7111};
int csr_N_range[] = {-3, 0, 842, 4441};

std::vector<double> csr_alpha_range = {2.0, 3.0};
std::vector<double> csr_beta_range  = {0.0, 1.0};

base csr_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csr_bin[] = {"rma10.bin",
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
                         "nos7.bin"};

class parameterized_csrmv : public testing::TestWithParam<csrmv_tuple>
{
    protected:
    parameterized_csrmv() {}
    virtual ~parameterized_csrmv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrmv_bin : public testing::TestWithParam<csrmv_bin_tuple>
{
    protected:
    parameterized_csrmv_bin() {}
    virtual ~parameterized_csrmv_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrmv_arguments(csrmv_tuple tup)
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

Arguments setup_csrmv_arguments(csrmv_bin_tuple tup)
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

TEST(csrmv_bad_arg, csrmv_float) { testing_csrmv_bad_arg<float>(); }

TEST_P(parameterized_csrmv, csrmv_float)
{
    Arguments arg = setup_csrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmv, csrmv_double)
{
    Arguments arg = setup_csrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmv_bin, csrmv_bin_float)
{
    Arguments arg = setup_csrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmv_bin, csrmv_bin_double)
{
    Arguments arg = setup_csrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(csrmv,
                        parameterized_csrmv,
                        testing::Combine(testing::ValuesIn(csr_M_range),
                                         testing::ValuesIn(csr_N_range),
                                         testing::ValuesIn(csr_alpha_range),
                                         testing::ValuesIn(csr_beta_range),
                                         testing::ValuesIn(csr_idxbase_range)));

INSTANTIATE_TEST_CASE_P(csrmv_bin,
                        parameterized_csrmv_bin,
                        testing::Combine(testing::ValuesIn(csr_alpha_range),
                                         testing::ValuesIn(csr_beta_range),
                                         testing::ValuesIn(csr_idxbase_range),
                                         testing::ValuesIn(csr_bin)));
