/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_doti.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef hipsparseIndexBase_t base;
typedef std::tuple<int, int, base> doti_tuple;

int doti_N_range[]   = {12000, 15332, 22031};
int doti_nnz_range[] = {-1, 0, 5, 10, 500, 1000, 7111, 10000};

base doti_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_doti : public testing::TestWithParam<doti_tuple>
{
    protected:
    parameterized_doti() {}
    virtual ~parameterized_doti() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_doti_arguments(doti_tuple tup)
{
    Arguments arg;
    arg.N        = std::get<0>(tup);
    arg.nnz      = std::get<1>(tup);
    arg.idx_base = std::get<2>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(doti_bad_arg, doti_float) { testing_doti_bad_arg<float>(); }

TEST_P(parameterized_doti, doti_float)
{
    Arguments arg = setup_doti_arguments(GetParam());

    hipsparseStatus_t status = testing_doti<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_doti, doti_double)
{
    Arguments arg = setup_doti_arguments(GetParam());

    hipsparseStatus_t status = testing_doti<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(doti,
                        parameterized_doti,
                        testing::Combine(testing::ValuesIn(doti_N_range),
                                         testing::ValuesIn(doti_nnz_range),
                                         testing::ValuesIn(doti_idx_base_range)));
