/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csr2hyb.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, hipsparseIndexBase_t, hipsparseHybPartition_t, int> csr2hyb_tuple;

int csr2hyb_M_range[] = {-1, 0, 10, 500, 872, 1000};
int csr2hyb_N_range[] = {-3, 0, 33, 242, 623, 1000};

hipsparseIndexBase_t csr2hyb_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO,
                                                 HIPSPARSE_INDEX_BASE_ONE};

hipsparseHybPartition_t csr2hyb_partition[] = {
    HIPSPARSE_HYB_PARTITION_AUTO, HIPSPARSE_HYB_PARTITION_MAX, HIPSPARSE_HYB_PARTITION_USER};

int csr2hyb_ELL_range[] = {-33, -1, 0, INT32_MAX};

class parameterized_csr2hyb : public testing::TestWithParam<csr2hyb_tuple>
{
    protected:
    parameterized_csr2hyb() {}
    virtual ~parameterized_csr2hyb() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csr2hyb_arguments(csr2hyb_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.idx_base  = std::get<2>(tup);
    arg.part      = std::get<3>(tup);
    arg.ell_width = std::get<4>(tup);
    arg.timing    = 0;
    return arg;
}

TEST(csr2hyb_bad_arg, csr2hyb) { testing_csr2hyb_bad_arg<float>(); }

TEST_P(parameterized_csr2hyb, csr2hyb_float)
{
    Arguments arg = setup_csr2hyb_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2hyb<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2hyb, csr2hyb_double)
{
    Arguments arg = setup_csr2hyb_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2hyb<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_CASE_P(csr2hyb,
                        parameterized_csr2hyb,
                        testing::Combine(testing::ValuesIn(csr2hyb_M_range),
                                         testing::ValuesIn(csr2hyb_N_range),
                                         testing::ValuesIn(csr2hyb_idx_base_range),
                                         testing::ValuesIn(csr2hyb_partition),
                                         testing::ValuesIn(csr2hyb_ELL_range)));
