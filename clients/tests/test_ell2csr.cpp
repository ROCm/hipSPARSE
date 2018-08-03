/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_ell2csr.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>

typedef std::tuple<int, int, hipsparseIndexBase_t, hipsparseIndexBase_t> ell2csr_tuple;
typedef std::tuple<hipsparseIndexBase_t, hipsparseIndexBase_t, std::string> ell2csr_bin_tuple;

int ell2csr_M_range[] = {-1, 0, 10, 500, 872, 1000};
int ell2csr_N_range[] = {-3, 0, 33, 242, 623, 1000};

hipsparseIndexBase_t ell2csr_ell_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO,
                                                 HIPSPARSE_INDEX_BASE_ONE};
hipsparseIndexBase_t ell2csr_csr_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO,
                                                 HIPSPARSE_INDEX_BASE_ONE};

std::string ell2csr_bin[] = {"rma10.bin",
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

class parameterized_ell2csr : public testing::TestWithParam<ell2csr_tuple>
{
    protected:
    parameterized_ell2csr() {}
    virtual ~parameterized_ell2csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_ell2csr_bin : public testing::TestWithParam<ell2csr_bin_tuple>
{
    protected:
    parameterized_ell2csr_bin() {}
    virtual ~parameterized_ell2csr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_ell2csr_arguments(ell2csr_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.idx_base  = std::get<2>(tup);
    arg.idx_base2 = std::get<3>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_ell2csr_arguments(ell2csr_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.N         = -99;
    arg.idx_base  = std::get<0>(tup);
    arg.idx_base2 = std::get<1>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<2>(tup);

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

TEST(ell2csr_bad_arg, ell2csr) { testing_ell2csr_bad_arg<float>(); }

TEST_P(parameterized_ell2csr, ell2csr_float)
{
    Arguments arg = setup_ell2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_ell2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_ell2csr, ell2csr_double)
{
    Arguments arg = setup_ell2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_ell2csr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_ell2csr_bin, ell2csr_bin_float)
{
    Arguments arg = setup_ell2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_ell2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_ell2csr_bin, ell2csr_bin_double)
{
    Arguments arg = setup_ell2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_ell2csr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(ell2csr,
                        parameterized_ell2csr,
                        testing::Combine(testing::ValuesIn(ell2csr_M_range),
                                         testing::ValuesIn(ell2csr_N_range),
                                         testing::ValuesIn(ell2csr_ell_base_range),
                                         testing::ValuesIn(ell2csr_csr_base_range)));

INSTANTIATE_TEST_CASE_P(ell2csr_bin,
                        parameterized_ell2csr_bin,
                        testing::Combine(testing::ValuesIn(ell2csr_ell_base_range),
                                         testing::ValuesIn(ell2csr_csr_base_range),
                                         testing::ValuesIn(ell2csr_bin)));
