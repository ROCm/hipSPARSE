/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csr2ell.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, hipsparseIndexBase_t, hipsparseIndexBase_t> csr2ell_tuple;

int csr2ell_M_range[] = {-1, 0, 10, 500, 872, 1000};
int csr2ell_N_range[] = {-3, 0, 33, 242, 623, 1000};

hipsparseIndexBase_t csr2ell_csr_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO,
                                                 HIPSPARSE_INDEX_BASE_ONE};
hipsparseIndexBase_t csr2ell_ell_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO,
                                                 HIPSPARSE_INDEX_BASE_ONE};

class parameterized_csr2ell : public testing::TestWithParam<csr2ell_tuple>
{
    protected:
    parameterized_csr2ell() {}
    virtual ~parameterized_csr2ell() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csr2ell_arguments(csr2ell_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.idx_base  = std::get<2>(tup);
    arg.idx_base2 = std::get<3>(tup);
    arg.timing    = 0;
    return arg;
}

TEST(csr2ell_bad_arg, csr2ell) { testing_csr2ell_bad_arg<float>(); }

TEST_P(parameterized_csr2ell, csr2ell_float)
{
    Arguments arg = setup_csr2ell_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2ell<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2ell, csr2ell_double)
{
    Arguments arg = setup_csr2ell_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2ell<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(csr2ell,
                        parameterized_csr2ell,
                        testing::Combine(testing::ValuesIn(csr2ell_M_range),
                                         testing::ValuesIn(csr2ell_N_range),
                                         testing::ValuesIn(csr2ell_csr_base_range),
                                         testing::ValuesIn(csr2ell_ell_base_range)));
