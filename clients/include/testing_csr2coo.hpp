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
#ifndef TESTING_CSR2COO_HPP
#define TESTING_CSR2COO_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_csr2coo_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int               m         = 100;
    int               nnz       = 100;
    int               safe_size = 100;
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto coo_row_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};

    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* coo_row_ind = (int*)coo_row_ind_managed.get();

    if(!csr_row_ptr || !coo_row_ind)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing for(csr_row_ptr == nullptr)
    {
        int* csr_row_ptr_null = nullptr;

        status = hipsparseXcsr2coo(
            handle, csr_row_ptr_null, nnz, m, coo_row_ind, HIPSPARSE_INDEX_BASE_ZERO);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }
    // Testing for(coo_row_ind == nullptr)
    {
        int* coo_row_ind_null = nullptr;

        status = hipsparseXcsr2coo(
            handle, csr_row_ptr, nnz, m, coo_row_ind_null, HIPSPARSE_INDEX_BASE_ZERO);
        verify_hipsparse_status_invalid_pointer(status, "Error: coo_row_ind is nullptr");
    }
    // Testing for(handle == nullptr)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsr2coo(
            handle_null, csr_row_ptr, nnz, m, coo_row_ind, HIPSPARSE_INDEX_BASE_ZERO);
        verify_hipsparse_status_invalid_handle(status);
    }
#endif
}

hipsparseStatus_t testing_csr2coo(Arguments argus)
{
    int                  m        = argus.M;
    int                  n        = argus.N;
    hipsparseIndexBase_t idx_base = argus.idx_base;
    std::string          filename = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    srand(12345ULL);

    // Host structures
    std::vector<int>   hcsr_row_ptr;
    std::vector<int>   hcsr_col_ind;
    std::vector<float> hcsr_val;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_csr_matrix(filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // Allocate memory on the device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcoo_row_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcoo_row_ind = (int*)dcoo_row_ind_managed.get();

    // Copy data from host to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(
            hipsparseXcsr2coo(handle, dcsr_row_ptr, nnz, m, dcoo_row_ind, idx_base));

        // Copy output from device to host
        std::vector<int> hcoo_row_ind(nnz);
        CHECK_HIP_ERROR(
            hipMemcpy(hcoo_row_ind.data(), dcoo_row_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));

        // CPU conversion to COO
        std::vector<int> hcoo_row_ind_gold(nnz);
        for(int i = 0; i < m; ++i)
        {
            int row_begin = hcsr_row_ptr[i] - idx_base;
            int row_end   = hcsr_row_ptr[i + 1] - idx_base;

            for(int j = row_begin; j < row_end; ++j)
            {
                hcoo_row_ind_gold[j] = i + idx_base;
            }
        }

        // Unit check
        unit_check_general(1, nnz, 1, hcoo_row_ind_gold.data(), hcoo_row_ind.data());
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSR2COO_HPP
