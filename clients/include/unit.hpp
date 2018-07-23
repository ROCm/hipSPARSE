/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef UNIT_HPP
#define UNIT_HPP

#include <hipsparse.h>

/* =====================================================================

    Google Unit check: ASSERT_EQ( elementof(A), elementof(B))

   =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Google Unit check.
 */

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going
template <typename T>
void unit_check_general(int M, int N, T* hCPU, T* hGPU);

#endif // UNIT_HPP
