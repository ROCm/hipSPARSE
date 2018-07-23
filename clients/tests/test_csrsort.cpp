/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrsort.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, int, hipsparseIndexBase_t> csrsort_tuple;

int csrsort_M_range[]               = {-1, 0, 10, 500, 872, 1000};
int csrsort_N_range[]               = {-3, 0, 33, 242, 623, 1000};
int csrsort_perm[]                  = {0, 1};
hipsparseIndexBase_t csrsort_base[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_csrsort : public testing::TestWithParam<csrsort_tuple>
{
    protected:
    parameterized_csrsort() {}
    virtual ~parameterized_csrsort() {}
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

TEST(csrsort_bad_arg, csrsort) { testing_csrsort_bad_arg(); }

TEST_P(parameterized_csrsort, csrsort)
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
