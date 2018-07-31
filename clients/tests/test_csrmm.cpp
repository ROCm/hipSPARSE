/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrmm.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <string>

typedef hipsparseIndexBase_t base;
typedef hipsparseOperation_t trans;
typedef std::tuple<int, int, int, double, double, base, trans, trans> csrmm_tuple;
typedef std::tuple<int, double, double, base, trans, trans, std::string> csrmm_bin_tuple;

int csrmm_M_range[] = {-1, 0, 42, 511, 3521};
int csrmm_N_range[] = {-1, 0, 13, 33, 64, 73};
int csrmm_K_range[] = {-1, 0, 50, 254, 1942};

double csrmm_alpha_range[] = {-1.0, 0.0, 3.3};
double csrmm_beta_range[]  = {-0.3, 0.0, 1.0};

base csrmm_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
trans csrmm_transA_range[] = {HIPSPARSE_OPERATION_NON_TRANSPOSE};
trans csrmm_transB_range[] = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};

std::string csrmm_bin[] = {"rma10.bin",
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

class parameterized_csrmm : public testing::TestWithParam<csrmm_tuple>
{
    protected:
    parameterized_csrmm() {}
    virtual ~parameterized_csrmm() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrmm_bin : public testing::TestWithParam<csrmm_bin_tuple>
{
    protected:
    parameterized_csrmm_bin() {}
    virtual ~parameterized_csrmm_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrmm_arguments(csrmm_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.K        = std::get<2>(tup);
    arg.alpha    = std::get<3>(tup);
    arg.beta     = std::get<4>(tup);
    arg.idx_base = std::get<5>(tup);
    arg.transA   = std::get<6>(tup);
    arg.transB   = std::get<7>(tup);
    arg.timing   = 0;
    return arg;
}

Arguments setup_csrmm_arguments(csrmm_bin_tuple tup)
{
    Arguments arg;
    arg.M        = -99;
    arg.N        = std::get<0>(tup);
    arg.K        = -99;
    arg.alpha    = std::get<1>(tup);
    arg.beta     = std::get<2>(tup);
    arg.idx_base = std::get<3>(tup);
    arg.transA   = std::get<4>(tup);
    arg.transB   = std::get<5>(tup);
    arg.timing   = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<6>(tup);

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

TEST(csrmm_bad_arg, csrmm_float) { testing_csrmm_bad_arg<float>(); }

TEST_P(parameterized_csrmm, csrmm_float)
{
    Arguments arg = setup_csrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmm<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmm, csrmm_double)
{
    Arguments arg = setup_csrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmm<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmm_bin, csrmm_bin_float)
{
    Arguments arg = setup_csrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmm<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmm_bin, csrmm_bin_double)
{
    Arguments arg = setup_csrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmm<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(csrmm,
                        parameterized_csrmm,
                        testing::Combine(testing::ValuesIn(csrmm_M_range),
                                         testing::ValuesIn(csrmm_N_range),
                                         testing::ValuesIn(csrmm_K_range),
                                         testing::ValuesIn(csrmm_alpha_range),
                                         testing::ValuesIn(csrmm_beta_range),
                                         testing::ValuesIn(csrmm_idxbase_range),
                                         testing::ValuesIn(csrmm_transA_range),
                                         testing::ValuesIn(csrmm_transB_range)));

INSTANTIATE_TEST_CASE_P(csrmm_bin,
                        parameterized_csrmm_bin,
                        testing::Combine(testing::ValuesIn(csrmm_N_range),
                                         testing::ValuesIn(csrmm_alpha_range),
                                         testing::ValuesIn(csrmm_beta_range),
                                         testing::ValuesIn(csrmm_idxbase_range),
                                         testing::ValuesIn(csrmm_transA_range),
                                         testing::ValuesIn(csrmm_transB_range),
                                         testing::ValuesIn(csrmm_bin)));
