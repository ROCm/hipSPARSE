/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrilusv.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <unistd.h>
#include <vector>
#include <string>

typedef hipsparseIndexBase_t base;

typedef std::tuple<base, std::string> csrilusv_bin_tuple;

base csrilusv_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csrilusv_bin[] = {"mac_econ_fwd500.bin",
                              "mc2depi.bin",
                              "scircuit.bin",
#if defined(__HIP_PLATFORM_HCC__)
                              "bmwcra_1.bin",
                              "nos1.bin",
#endif
                              "nos3.bin",
                              "nos4.bin",
                              "nos5.bin",
                              "nos6.bin",
                              "nos7.bin"};

class parameterized_csrilusv_bin : public testing::TestWithParam<csrilusv_bin_tuple>
{
    protected:
    parameterized_csrilusv_bin() {}
    virtual ~parameterized_csrilusv_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrilusv_arguments(csrilusv_bin_tuple tup)
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

TEST_P(parameterized_csrilusv_bin, csrilusv_bin_float)
{
    Arguments arg = setup_csrilusv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilusv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrilusv_bin, csrilusv_bin_double)
{
    Arguments arg = setup_csrilusv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilusv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(csrilusv_bin,
                        parameterized_csrilusv_bin,
                        testing::Combine(testing::ValuesIn(csrilusv_idxbase_range),
                                         testing::ValuesIn(csrilusv_bin)));
