/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_coo2csr.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, hipsparseIndexBase_t> coo2csr_tuple;

int coo2csr_M_range[] = {-1, 0, 10, 500, 872, 1000};
int coo2csr_N_range[] = {-3, 0, 33, 242, 623, 1000};

hipsparseIndexBase_t coo2csr_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO,
                                                 HIPSPARSE_INDEX_BASE_ONE};

class parameterized_coo2csr : public testing::TestWithParam<coo2csr_tuple>
{
    protected:
    parameterized_coo2csr() {}
    virtual ~parameterized_coo2csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_coo2csr_arguments(coo2csr_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.idx_base = std::get<2>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(coo2csr_bad_arg, coo2csr) { testing_coo2csr_bad_arg(); }

TEST_P(parameterized_coo2csr, coo2csr)
{
    Arguments arg = setup_coo2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_coo2csr(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(coo2csr,
                        parameterized_coo2csr,
                        testing::Combine(testing::ValuesIn(coo2csr_M_range),
                                         testing::ValuesIn(coo2csr_N_range),
                                         testing::ValuesIn(coo2csr_idx_base_range)));
