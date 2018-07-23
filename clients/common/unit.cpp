/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "unit.hpp"

#include <hipsparse.h>
#include <hip/hip_runtime_api.h>

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#else
#ifdef NDEBUG
#undef NDEBUG
#include <assert.h>
#define NDEBUG
#else
#include <assert.h>
#endif
#endif

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going

template <>
void unit_check_general(int M, int N, float* hCPU, float* hGPU)
{
    for(int j = 0; j < N; j++)
    {
        for(int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_FLOAT_EQ(hCPU[i + j], hGPU[i + j]);
#else
            assert(hCPU[i + j] == hGPU[i + j]);
#endif
        }
    }
}

template <>
void unit_check_general(int M, int N, double* hCPU, double* hGPU)
{
    for(int j = 0; j < N; j++)
    {
        for(int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_DOUBLE_EQ(hCPU[i + j], hGPU[i + j]);
#else
            assert(hCPU[i + j] == hGPU[i + j]);
#endif
        }
    }
}

template <>
void unit_check_general(int M, int N, int* hCPU, int* hGPU)
{
    for(int j = 0; j < N; j++)
    {
        for(int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_EQ(hCPU[i + j], hGPU[i + j]);
#else
            assert(hCPU[i + j] == hGPU[i + j]);
#endif
        }
    }
}

template <>
void unit_check_general(int M, int N, size_t* hCPU, size_t* hGPU)
{
    for(int j = 0; j < N; j++)
    {
        for(int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_EQ(hCPU[i + j], hGPU[i + j]);
#else
            assert(hCPU[i + j] == hGPU[i + j]);
#endif
        }
    }
}
