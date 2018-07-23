/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_coomv.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef hipsparseIndexBase_t base;
typedef std::tuple<int, int, double, double, base> coomv_tuple;

int coo_M_range[] = {-1, 0, 10, 500, 7111, 10000};
int coo_N_range[] = {-3, 0, 33, 842, 4441, 10000};

std::vector<double> coo_alpha_range = {2.0, 3.0};
std::vector<double> coo_beta_range  = {0.0, 0.67, 1.0};

base coo_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_coomv : public testing::TestWithParam<coomv_tuple>
{
    protected:
    parameterized_coomv() {}
    virtual ~parameterized_coomv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_coomv_arguments(coomv_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.alpha    = std::get<2>(tup);
    arg.beta     = std::get<3>(tup);
    arg.idx_base = std::get<4>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(coomv_bad_arg, coomv_float) { testing_coomv_bad_arg<float>(); }

TEST_P(parameterized_coomv, coomv_float)
{
    Arguments arg = setup_coomv_arguments(GetParam());

    hipsparseStatus_t status = testing_coomv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_coomv, coomv_double)
{
    Arguments arg = setup_coomv_arguments(GetParam());

    hipsparseStatus_t status = testing_coomv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(coomv,
                        parameterized_coomv,
                        testing::Combine(testing::ValuesIn(coo_M_range),
                                         testing::ValuesIn(coo_N_range),
                                         testing::ValuesIn(coo_alpha_range),
                                         testing::ValuesIn(coo_beta_range),
                                         testing::ValuesIn(coo_idxbase_range)));
