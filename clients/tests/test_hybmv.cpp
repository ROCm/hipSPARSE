/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_hybmv.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>

typedef std::tuple<int, int, double, double, hipsparseIndexBase_t, hipsparseHybPartition_t, int>
    hybmv_tuple;
typedef std::tuple<double, double, hipsparseIndexBase_t, hipsparseHybPartition_t, int, std::string>
    hybmv_bin_tuple;

int hyb_M_range[] = {-1, 0, 10, 500, 7111, 10000};
int hyb_N_range[] = {-3, 0, 33, 842, 4441, 10000};

std::vector<double> hyb_alpha_range = {2.0, 3.0};
std::vector<double> hyb_beta_range  = {0.0, 0.67, 1.0};

hipsparseIndexBase_t hyb_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

hipsparseHybPartition_t hyb_partition[] = {
    HIPSPARSE_HYB_PARTITION_AUTO, HIPSPARSE_HYB_PARTITION_MAX, HIPSPARSE_HYB_PARTITION_USER};

int hyb_ELL_range[] = {0, 1, 2};

std::string hyb_bin[] = {"rma10.bin",
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

class parameterized_hybmv : public testing::TestWithParam<hybmv_tuple>
{
    protected:
    parameterized_hybmv() {}
    virtual ~parameterized_hybmv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_hybmv_bin : public testing::TestWithParam<hybmv_bin_tuple>
{
    protected:
    parameterized_hybmv_bin() {}
    virtual ~parameterized_hybmv_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_hybmv_arguments(hybmv_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.alpha     = std::get<2>(tup);
    arg.beta      = std::get<3>(tup);
    arg.idx_base  = std::get<4>(tup);
    arg.part      = std::get<5>(tup);
    arg.ell_width = std::get<6>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_hybmv_arguments(hybmv_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.N         = -99;
    arg.alpha     = std::get<0>(tup);
    arg.beta      = std::get<1>(tup);
    arg.idx_base  = std::get<2>(tup);
    arg.part      = std::get<3>(tup);
    arg.ell_width = std::get<4>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<5>(tup);

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

TEST(hybmv_bad_arg, hybmv_float) { testing_hybmv_bad_arg<float>(); }

TEST_P(parameterized_hybmv, hybmv_float)
{
    Arguments arg = setup_hybmv_arguments(GetParam());

    hipsparseStatus_t status = testing_hybmv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_hybmv, hybmv_double)
{
    Arguments arg = setup_hybmv_arguments(GetParam());

    hipsparseStatus_t status = testing_hybmv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_hybmv_bin, hybmv_bin_float)
{
    Arguments arg = setup_hybmv_arguments(GetParam());

    hipsparseStatus_t status = testing_hybmv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_hybmv_bin, hybmv_bin_double)
{
    Arguments arg = setup_hybmv_arguments(GetParam());

    hipsparseStatus_t status = testing_hybmv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(hybmv,
                        parameterized_hybmv,
                        testing::Combine(testing::ValuesIn(hyb_M_range),
                                         testing::ValuesIn(hyb_N_range),
                                         testing::ValuesIn(hyb_alpha_range),
                                         testing::ValuesIn(hyb_beta_range),
                                         testing::ValuesIn(hyb_idxbase_range),
                                         testing::ValuesIn(hyb_partition),
                                         testing::ValuesIn(hyb_ELL_range)));

INSTANTIATE_TEST_CASE_P(hybmv_bin,
                        parameterized_hybmv_bin,
                        testing::Combine(testing::ValuesIn(hyb_alpha_range),
                                         testing::ValuesIn(hyb_beta_range),
                                         testing::ValuesIn(hyb_idxbase_range),
                                         testing::ValuesIn(hyb_partition),
                                         testing::ValuesIn(hyb_ELL_range),
                                         testing::ValuesIn(hyb_bin)));
