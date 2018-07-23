/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_axpyi.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef hipsparseIndexBase_t base;
typedef std::tuple<int, int, double, base> axpyi_tuple;

int axpyi_N_range[]   = {12000, 15332, 22031};
int axpyi_nnz_range[] = {-1, 0, 5, 10, 500, 1000, 7111, 10000};

std::vector<double> axpyi_alpha_range = {1.0, 0.0};

base axpyi_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_axpyi : public testing::TestWithParam<axpyi_tuple>
{
    protected:
    parameterized_axpyi() {}
    virtual ~parameterized_axpyi() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_axpyi_arguments(axpyi_tuple tup)
{
    Arguments arg;
    arg.N        = std::get<0>(tup);
    arg.nnz      = std::get<1>(tup);
    arg.alpha    = std::get<2>(tup);
    arg.idx_base = std::get<3>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(axpyi_bad_arg, axpyi_float) { testing_axpyi_bad_arg<float>(); }

TEST_P(parameterized_axpyi, axpyi_float)
{
    Arguments arg = setup_axpyi_arguments(GetParam());

    hipsparseStatus_t status = testing_axpyi<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_axpyi, axpyi_double)
{
    Arguments arg = setup_axpyi_arguments(GetParam());

    hipsparseStatus_t status = testing_axpyi<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(axpyi,
                        parameterized_axpyi,
                        testing::Combine(testing::ValuesIn(axpyi_N_range),
                                         testing::ValuesIn(axpyi_nnz_range),
                                         testing::ValuesIn(axpyi_alpha_range),
                                         testing::ValuesIn(axpyi_idx_base_range)));
