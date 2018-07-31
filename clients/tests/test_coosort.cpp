/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_coosort.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, hipsparseOperation_t, int, hipsparseIndexBase_t> coosort_tuple;

int coosort_M_range[]               = {-1, 0, 10, 500, 3872, 10000};
int coosort_N_range[]               = {-3, 0, 33, 242, 1623, 10000};
hipsparseOperation_t coosort_trans[] = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
int coosort_perm[]                  = {0, 1};
hipsparseIndexBase_t coosort_base[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_coosort : public testing::TestWithParam<coosort_tuple>
{
    protected:
    parameterized_coosort() {}
    virtual ~parameterized_coosort() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_coosort_arguments(coosort_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.transA    = std::get<2>(tup);
    arg.temp     = std::get<3>(tup);
    arg.idx_base = std::get<4>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(coosort_bad_arg, coosort) { testing_coosort_bad_arg(); }

TEST_P(parameterized_coosort, coosort)
{
    Arguments arg = setup_coosort_arguments(GetParam());

    hipsparseStatus_t status = testing_coosort(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(coosort,
                        parameterized_coosort,
                        testing::Combine(testing::ValuesIn(coosort_M_range),
                                         testing::ValuesIn(coosort_N_range),
                                         testing::ValuesIn(coosort_trans),
                                         testing::ValuesIn(coosort_perm),
                                         testing::ValuesIn(coosort_base)));
