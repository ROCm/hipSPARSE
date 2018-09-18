/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef ARG_CHECK_HPP
#define ARG_CHECK_HPP

#include <hipsparse.h>

void verify_hipsparse_status_invalid_pointer(hipsparseStatus_t status, const char* message);

void verify_hipsparse_status_invalid_size(hipsparseStatus_t status, const char* message);

void verify_hipsparse_status_invalid_value(hipsparseStatus_t status, const char* message);

void verify_hipsparse_status_zero_pivot(hipsparseStatus_t status, const char* message);

void verify_hipsparse_status_invalid_handle(hipsparseStatus_t status);

void verify_hipsparse_status_success(hipsparseStatus_t status, const char* message);

#endif // ARG_CHECK_HPP
