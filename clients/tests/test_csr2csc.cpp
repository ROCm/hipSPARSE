/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csr2csc.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, hipsparseAction_t, hipsparseIndexBase_t> csr2csc_tuple;

int csr2csc_M_range[] = {-1, 0, 10, 500, 872, 1000};
int csr2csc_N_range[] = {-3, 0, 33, 242, 623, 1000};

hipsparseAction_t csr2csc_action_range[] = {HIPSPARSE_ACTION_NUMERIC, HIPSPARSE_ACTION_SYMBOLIC};

hipsparseIndexBase_t csr2csc_csr_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO,
                                                 HIPSPARSE_INDEX_BASE_ONE};

class parameterized_csr2csc : public testing::TestWithParam<csr2csc_tuple>
{
    protected:
    parameterized_csr2csc() {}
    virtual ~parameterized_csr2csc() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csr2csc_arguments(csr2csc_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.action   = std::get<2>(tup);
    arg.idx_base = std::get<3>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(csr2csc_bad_arg, csr2csc) { testing_csr2csc_bad_arg<float>(); }

TEST_P(parameterized_csr2csc, csr2csc_float)
{
    Arguments arg = setup_csr2csc_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csc<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2csc, csr2csc_double)
{
    Arguments arg = setup_csr2csc_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csc<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(csr2csc,
                        parameterized_csr2csc,
                        testing::Combine(testing::ValuesIn(csr2csc_M_range),
                                         testing::ValuesIn(csr2csc_N_range),
                                         testing::ValuesIn(csr2csc_action_range),
                                         testing::ValuesIn(csr2csc_csr_base_range)));
