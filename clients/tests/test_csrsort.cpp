/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrsort.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>

typedef std::tuple<int, int, int, hipsparseIndexBase_t> csrsort_tuple;
typedef std::tuple<int, hipsparseIndexBase_t, std::string> csrsort_bin_tuple;

int csrsort_M_range[]               = {-1, 0, 10, 500, 872, 1000};
int csrsort_N_range[]               = {-3, 0, 33, 242, 623, 1000};
int csrsort_perm[]                  = {0, 1};
hipsparseIndexBase_t csrsort_base[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csrsort_bin[] = {"rma10.bin",
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

class parameterized_csrsort : public testing::TestWithParam<csrsort_tuple>
{
    protected:
    parameterized_csrsort() {}
    virtual ~parameterized_csrsort() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrsort_bin : public testing::TestWithParam<csrsort_bin_tuple>
{
    protected:
    parameterized_csrsort_bin() {}
    virtual ~parameterized_csrsort_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrsort_arguments(csrsort_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.temp     = std::get<2>(tup);
    arg.idx_base = std::get<3>(tup);
    arg.timing   = 0;
    return arg;
}

Arguments setup_csrsort_arguments(csrsort_bin_tuple tup)
{
    Arguments arg;
    arg.M        = -99;
    arg.N        = -99;
    arg.temp     = std::get<0>(tup);
    arg.idx_base = std::get<1>(tup);
    arg.timing   = 0;

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

TEST(csrsort_bad_arg, csrsort) { testing_csrsort_bad_arg(); }

TEST_P(parameterized_csrsort, csrsort)
{
    Arguments arg = setup_csrsort_arguments(GetParam());

    hipsparseStatus_t status = testing_csrsort(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrsort_bin, csrsort_bin)
{
    Arguments arg = setup_csrsort_arguments(GetParam());

    hipsparseStatus_t status = testing_csrsort(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(csrsort,
                        parameterized_csrsort,
                        testing::Combine(testing::ValuesIn(csrsort_M_range),
                                         testing::ValuesIn(csrsort_N_range),
                                         testing::ValuesIn(csrsort_perm),
                                         testing::ValuesIn(csrsort_base)));

INSTANTIATE_TEST_CASE_P(csrsort_bin,
                        parameterized_csrsort_bin,
                        testing::Combine(testing::ValuesIn(csrsort_perm),
                                         testing::ValuesIn(csrsort_base),
                                         testing::ValuesIn(csrsort_bin)));
