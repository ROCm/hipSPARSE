/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "arg_check.hpp"

#include <iostream>
#include <hip/hip_runtime_api.h>
#include <hipsparse.h>

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                    \
        {                                                         \
            fprintf(stderr,                                       \
                    "hip error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                         \
                    __FILE__,                                     \
                    __LINE__);                                    \
        }                                                         \
    }

void verify_hipsparse_status_invalid_pointer(hipsparseStatus_t status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPSPARSE_STATUS_INVALID_VALUE);
#else
    if(status != HIPSPARSE_STATUS_INVALID_VALUE)
    {
        std::cerr << "rocSPARSE TEST ERROR: status != HIPSPARSE_STATUS_INVALID_VALUE, ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_hipsparse_status_invalid_size(hipsparseStatus_t status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPSPARSE_STATUS_INVALID_VALUE);
#else
    if(status != HIPSPARSE_STATUS_INVALID_VALUE)
    {
        std::cerr << "rocSPARSE TEST ERROR: status != HIPSPARSE_STATUS_INVALID_VALUE, ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_hipsparse_status_invalid_value(hipsparseStatus_t status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPSPARSE_STATUS_INVALID_VALUE);
#else
    if(status != HIPSPARSE_STATUS_INVALID_VALUE)
    {
        std::cerr << "rocSPARSE TEST ERROR: status != HIPSPARSE_STATUS_INVALID_VALUE, ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_hipsparse_status_invalid_handle(hipsparseStatus_t status)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPSPARSE_STATUS_NOT_INITIALIZED);
#else
    if(status != HIPSPARSE_STATUS_NOT_INITIALIZED)
    {
        std::cerr << "rocSPARSE TEST ERROR: status != HIPSPARSE_STATUS_NOT_INITIALIZED"
                  << std::endl;
    }
#endif
}

void verify_hipsparse_status_success(hipsparseStatus_t status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
#else
    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        std::cerr << "rocSPARSE TEST ERROR: status != HIPSPARSE_STATUS_SUCCESS, ";
        std::cerr << message << std::endl;
    }
#endif
}
