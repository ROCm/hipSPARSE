/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#ifndef TESTING_COOSORT_HPP
#define TESTING_COOSORT_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_coosort_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVCC__
    // do not test for bad args
    return;
#endif
    int               m         = 100;
    int               n         = 100;
    int               nnz       = 100;
    int               safe_size = 100;
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    size_t buffer_size = 0;

    auto coo_row_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto coo_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto perm_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto buffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*  coo_row_ind = (int*)coo_row_ind_managed.get();
    int*  coo_col_ind = (int*)coo_col_ind_managed.get();
    int*  perm        = (int*)perm_managed.get();
    void* buffer      = (void*)buffer_managed.get();

    if(!coo_row_ind || !coo_col_ind || !perm || !buffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing coosort_buffer_size for bad args

    // Testing for (coo_row_ind == nullptr)
    {
        int* coo_row_ind_null = nullptr;

        status = hipsparseXcoosort_bufferSizeExt(
            handle, m, n, nnz, coo_row_ind_null, coo_col_ind, &buffer_size);
        verify_hipsparse_status_invalid_pointer(status, "Error: coo_row_ind is nullptr");
    }

    // Testing for (coo_col_ind == nullptr)
    {
        int* coo_col_ind_null = nullptr;

        status = hipsparseXcoosort_bufferSizeExt(
            handle, m, n, nnz, coo_row_ind, coo_col_ind_null, &buffer_size);
        verify_hipsparse_status_invalid_pointer(status, "Error: coo_col_ind is nullptr");
    }

    // Testing for (buffer_size == nullptr)
    {
        size_t* buffer_size_null = nullptr;

        status = hipsparseXcoosort_bufferSizeExt(
            handle, m, n, nnz, coo_row_ind, coo_col_ind, buffer_size_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: buffer_size is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcoosort_bufferSizeExt(
            handle_null, m, n, nnz, coo_row_ind, coo_col_ind, &buffer_size);
        verify_hipsparse_status_invalid_handle(status);
    }

    // Testing coosort_by_row for bad args

    // Testing for (coo_row_ind == nullptr)
    {
        int* coo_row_ind_null = nullptr;

        status = hipsparseXcoosortByRow(
            handle, m, n, nnz, coo_row_ind_null, coo_col_ind, perm, buffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: coo_row_ind is nullptr");
    }

    // Testing for (coo_col_ind == nullptr)
    {
        int* coo_col_ind_null = nullptr;

        status = hipsparseXcoosortByRow(
            handle, m, n, nnz, coo_row_ind, coo_col_ind_null, perm, buffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: coo_col_ind is nullptr");
    }

    // Testing for (buffer == nullptr)
    {
        int* buffer_null = nullptr;

        status = hipsparseXcoosortByRow(
            handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, buffer_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: buffer is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcoosortByRow(
            handle_null, m, n, nnz, coo_row_ind, coo_col_ind, perm, buffer);
        verify_hipsparse_status_invalid_handle(status);
    }

    // Testing coosort_by_column for bad args

    // Testing for (coo_row_ind == nullptr)
    {
        int* coo_row_ind_null = nullptr;

        status = hipsparseXcoosortByColumn(
            handle, m, n, nnz, coo_row_ind_null, coo_col_ind, perm, buffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: coo_row_ind is nullptr");
    }

    // Testing for (coo_col_ind == nullptr)
    {
        int* coo_col_ind_null = nullptr;

        status = hipsparseXcoosortByColumn(
            handle, m, n, nnz, coo_row_ind, coo_col_ind_null, perm, buffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: coo_col_ind is nullptr");
    }

    // Testing for (buffer == nullptr)
    {
        int* buffer_null = nullptr;

        status = hipsparseXcoosortByColumn(
            handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, buffer_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: buffer is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcoosortByColumn(
            handle_null, m, n, nnz, coo_row_ind, coo_col_ind, perm, buffer);
        verify_hipsparse_status_invalid_handle(status);
    }
}

hipsparseStatus_t testing_coosort(Arguments argus)
{
    int                  m         = argus.M;
    int                  n         = argus.N;
    int                  safe_size = 100;
    int                  by_row    = argus.transA == HIPSPARSE_OPERATION_NON_TRANSPOSE;
    int                  permute   = argus.temp;
    hipsparseIndexBase_t idx_base  = argus.idx_base;
    std::string          binfile   = "";
    std::string          filename  = "";
    hipsparseStatus_t    status;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    if(m == -99 && n == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        m = n = safe_size;
    }

    if(argus.timing == 1)
    {
        filename = argus.filename;
    }

    size_t buffer_size = 0;

    double scale = 0.02;
    if(m > 1000 || n > 1000)
    {
        scale = 2.0 / std::max(m, n);
    }
    int nnz = m * scale * n;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0)
    {
#ifdef __HIP_PLATFORM_NVCC__
        // Do not test args in cusparse
        return HIPSPARSE_STATUS_SUCCESS;
#endif
        auto coo_row_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto coo_col_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto perm_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto buffer_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

        int*  coo_row_ind = (int*)coo_row_ind_managed.get();
        int*  coo_col_ind = (int*)coo_col_ind_managed.get();
        int*  perm        = (int*)perm_managed.get();
        void* buffer      = (void*)buffer_managed.get();

        if(!coo_row_ind || !coo_col_ind || !perm || !buffer)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!coo_row_ind || !coo_col_ind || !perm || !buffer");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        status = hipsparseXcoosort_bufferSizeExt(
            handle, m, n, nnz, coo_row_ind, coo_col_ind, &buffer_size);

        if(m < 0 || n < 0 || nnz < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || nnz < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0");

            // Buffer size should be zero
            size_t zero = 0;
            unit_check_general(1, 1, 1, &zero, &buffer_size);
        }

        if(by_row)
        {
            status
                = hipsparseXcoosortByRow(handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, buffer);
        }
        else
        {
            status = hipsparseXcoosortByColumn(
                handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, buffer);
        }

        if(m < 0 || n < 0 || nnz < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || nnz < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // For testing, assemble a COO matrix and convert it to CSR first (on host)

    // Host structures
    std::vector<int>   hcoo_row_ind;
    std::vector<int>   hcoo_col_ind;
    std::vector<float> hcoo_val;

    // Sample initial COO matrix on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        std::vector<int> hcsr_row_ptr;
        if(read_bin_matrix(
               binfile.c_str(), m, n, nnz, hcsr_row_ptr, hcoo_col_ind, hcoo_val, idx_base)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }

        // Convert CSR to COO
        hcoo_row_ind.resize(nnz);
        for(int i = 0; i < m; ++i)
        {
            for(int j = hcsr_row_ptr[i]; j < hcsr_row_ptr[i + 1]; ++j)
            {
                hcoo_row_ind[j - idx_base] = i + idx_base;
            }
        }
    }
    else if(argus.laplacian)
    {
        std::vector<int> hcsr_row_ptr;
        m = n = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcoo_col_ind, hcoo_val, idx_base);
        nnz   = hcsr_row_ptr[m];

        // Convert CSR to COO
        hcoo_row_ind.resize(nnz);
        for(int i = 0; i < m; ++i)
        {
            for(int j = hcsr_row_ptr[i]; j < hcsr_row_ptr[i + 1]; ++j)
            {
                hcoo_row_ind[j - idx_base] = i + idx_base;
            }
        }
    }
    else
    {
        if(filename != "")
        {
            if(read_mtx_matrix(
                   filename.c_str(), m, n, nnz, hcoo_row_ind, hcoo_col_ind, hcoo_val, idx_base)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            gen_matrix_coo(m, n, nnz, hcoo_row_ind, hcoo_col_ind, hcoo_val, idx_base);
        }
    }

    // Unsort COO columns
    std::vector<int>   hcoo_row_ind_unsorted(nnz);
    std::vector<int>   hcoo_col_ind_unsorted(nnz);
    std::vector<float> hcoo_val_unsorted(nnz);

    hcoo_row_ind_unsorted = hcoo_row_ind;
    hcoo_col_ind_unsorted = hcoo_col_ind;
    hcoo_val_unsorted     = hcoo_val;

    for(int i = 0; i < nnz; ++i)
    {
        int rng = rand() % nnz;

        int   temp_row = hcoo_row_ind_unsorted[i];
        int   temp_col = hcoo_col_ind_unsorted[i];
        float temp_val = hcoo_val_unsorted[i];

        hcoo_row_ind_unsorted[i] = hcoo_row_ind_unsorted[rng];
        hcoo_col_ind_unsorted[i] = hcoo_col_ind_unsorted[rng];
        hcoo_val_unsorted[i]     = hcoo_val_unsorted[rng];

        hcoo_row_ind_unsorted[rng] = temp_row;
        hcoo_col_ind_unsorted[rng] = temp_col;
        hcoo_val_unsorted[rng]     = temp_val;
    }

    // If coosort by column, sort host arrays by column
    if(!by_row)
    {
        std::vector<int> hperm(nnz);
        for(int i = 0; i < nnz; ++i)
        {
            hperm[i] = i;
        }

        std::sort(hperm.begin(), hperm.end(), [&](const int& a, const int& b) {
            if(hcoo_col_ind_unsorted[a] < hcoo_col_ind_unsorted[b])
            {
                return true;
            }
            else if(hcoo_col_ind_unsorted[a] == hcoo_col_ind_unsorted[b])
            {
                return (hcoo_row_ind_unsorted[a] < hcoo_row_ind_unsorted[b]);
            }
            else
            {
                return false;
            }
        });

        for(int i = 0; i < nnz; ++i)
        {
            hcoo_row_ind[i] = hcoo_row_ind_unsorted[hperm[i]];
            hcoo_col_ind[i] = hcoo_col_ind_unsorted[hperm[i]];
            hcoo_val[i]     = hcoo_val_unsorted[hperm[i]];
        }
    }

    // Allocate memory on the device
    auto dcoo_row_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcoo_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcoo_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dcoo_val_sorted_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dperm_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};

    int*   dcoo_row_ind    = (int*)dcoo_row_ind_managed.get();
    int*   dcoo_col_ind    = (int*)dcoo_col_ind_managed.get();
    float* dcoo_val        = (float*)dcoo_val_managed.get();
    float* dcoo_val_sorted = (float*)dcoo_val_sorted_managed.get();

    // Set permutation vector, if asked for
    int* dperm = permute ? (int*)dperm_managed.get() : nullptr;

    if(!dcoo_row_ind || !dcoo_col_ind || !dcoo_val || !dcoo_val_sorted || (permute && !dperm))
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dcoo_row_ind || !dcoo_col_ind || !dcoo_val || "
                                        "!dcoo_val_sorted || (permute && !dperm)");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcoo_row_ind, hcoo_row_ind_unsorted.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcoo_col_ind, hcoo_col_ind_unsorted.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_val, hcoo_val_unsorted.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        // Obtain buffer size
        CHECK_HIPSPARSE_ERROR(hipsparseXcoosort_bufferSizeExt(
            handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, &buffer_size));

        // Allocate buffer on the device
        auto dbuffer_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(char) * buffer_size), device_free};

        void* dbuffer = (void*)dbuffer_managed.get();

        if(!dbuffer)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED, "!dbuffer");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        if(permute)
        {
            // Initialize perm with identity permutation
            CHECK_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, nnz, dperm));
        }

        // Sort CSR columns
        if(by_row)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcoosortByRow(
                handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, dperm, dbuffer));
        }
        else
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcoosortByColumn(
                handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, dperm, dbuffer));
        }

        if(permute)
        {
            // Sort CSR values
            CHECK_HIPSPARSE_ERROR(hipsparseSgthr(
                handle, nnz, dcoo_val, dcoo_val_sorted, dperm, HIPSPARSE_INDEX_BASE_ZERO));
        }

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcoo_row_ind_unsorted.data(), dcoo_row_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcoo_col_ind_unsorted.data(), dcoo_col_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));

        if(permute)
        {
            CHECK_HIP_ERROR(hipMemcpy(hcoo_val_unsorted.data(),
                                      dcoo_val_sorted,
                                      sizeof(float) * nnz,
                                      hipMemcpyDeviceToHost));
        }

        // Unit check
        unit_check_general(1, nnz, 1, hcoo_row_ind.data(), hcoo_row_ind_unsorted.data());
        unit_check_general(1, nnz, 1, hcoo_col_ind.data(), hcoo_col_ind_unsorted.data());

        if(permute)
        {
            unit_check_general(1, nnz, 1, hcoo_val.data(), hcoo_val_unsorted.data());
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Allocate buffer for coosort
        hipsparseXcoosort_bufferSizeExt(
            handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, &buffer_size);

        auto dbuffer_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(char) * buffer_size), device_free};
        void* dbuffer = (void*)dbuffer_managed.get();

        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            hipsparseXcoosortByRow(handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, nullptr, dbuffer);
        }

        double gpu_time_used = get_time_us();

        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            hipsparseXcoosortByRow(handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, nullptr, dbuffer);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        printf("m\t\tn\t\tnnz\t\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\n", m, n, nnz, gpu_time_used);
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_COOSORT_HPP
