/* ************************************************************************
 * Copyright (C) 2018-2019 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef TESTING_COO2CSR_HPP
#define TESTING_COO2CSR_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include "hipsparse_arguments.hpp"

#include <algorithm>
#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_coo2csr_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int               m         = 100;
    int               nnz       = 100;
    int               safe_size = 100;
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto coo_row_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};

    int* coo_row_ind = (int*)coo_row_ind_managed.get();
    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();

    if(!coo_row_ind || !csr_row_ptr)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing for(coo_row_ind == nullptr)
    {
        int* coo_row_ind_null = nullptr;

        status = hipsparseXcoo2csr(
            handle, coo_row_ind_null, nnz, m, csr_row_ptr, HIPSPARSE_INDEX_BASE_ZERO);
        verify_hipsparse_status_invalid_pointer(status, "Error: coo_row_ind is nullptr");
    }
    // Testing for(csr_row_ptr == nullptr)
    {
        int* csr_row_ptr_null = nullptr;

        status = hipsparseXcoo2csr(
            handle, coo_row_ind, nnz, m, csr_row_ptr_null, HIPSPARSE_INDEX_BASE_ZERO);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }
    // Testing for(handle == nullptr)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcoo2csr(
            handle_null, coo_row_ind, nnz, m, csr_row_ptr, HIPSPARSE_INDEX_BASE_ZERO);
        verify_hipsparse_status_invalid_handle(status);
    }
#endif
}

template<typename T>
hipsparseStatus_t testing_coo2csr(Arguments argus)
{
    int                  m        = argus.M;
    int                  n        = argus.N;
    hipsparseIndexBase_t idx_base = argus.baseA;
    std::string          filename = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    if(m == 0 || n == 0)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        return HIPSPARSE_STATUS_SUCCESS;
#endif
    }

    srand(12345ULL);

    // Host structures
    std::vector<int>   hcoo_row_ind;
    std::vector<int>   hcoo_col_ind;
    std::vector<T> hcoo_val;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_coo_matrix(filename, m, n, nnz, hcoo_row_ind, hcoo_col_ind, hcoo_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    std::vector<int> hcsr_row_ptr(m + 1);
    std::vector<int> hcsr_row_ptr_gold(m + 1, 0);

    // Allocate memory on the device
    auto dcoo_row_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};

    int* dcoo_row_ind = (int*)dcoo_row_ind_managed.get();
    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();

    // Copy data from host to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_row_ind, hcoo_row_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(
            hipsparseXcoo2csr(handle, dcoo_row_ind, nnz, m, dcsr_row_ptr, idx_base));

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr.data(), dcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));

        // CPU
        // coo2csr on host
        for(int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptr_gold[hcoo_row_ind[i] + 1 - idx_base];
        }

        hcsr_row_ptr_gold[0] = idx_base;
        for(int i = 0; i < m; ++i)
        {
            hcsr_row_ptr_gold[i + 1] += hcsr_row_ptr_gold[i];
        }

        // Unit check
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_gold.data(), hcsr_row_ptr.data());
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_COO2CSR_HPP
