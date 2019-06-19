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
#ifdef __HIP_PLATFORM_NVCC__
    // do not test for bad args
    return;
#endif
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
}

hipsparseStatus_t testing_csr2coo(Arguments argus)
{
    int                  m         = argus.M;
    int                  n         = argus.N;
    int                  safe_size = 100;
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
        auto csr_row_ptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto coo_row_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};

        int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
        int* coo_row_ind = (int*)coo_row_ind_managed.get();

        if(!csr_row_ptr || !coo_row_ind)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!csr_row_ptr || !coo_row_ind");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        status = hipsparseXcsr2coo(handle, csr_row_ptr, nnz, m, coo_row_ind, idx_base);

        if(m < 0 || nnz < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || nnz < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Host structures
    std::vector<int>   hcsr_row_ptr;
    std::vector<int>   hcoo_row_ind;
    std::vector<int>   hcol_ind;
    std::vector<float> hval(nnz);

    // Initial data on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(binfile.c_str(), m, n, nnz, hcsr_row_ptr, hcol_ind, hval, idx_base) != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }
    else if(argus.laplacian)
    {
        m = n = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcol_ind, hval, idx_base);
        nnz   = hcsr_row_ptr[m];
    }
    else
    {
        if(filename != "")
        {
            if(read_mtx_matrix(filename.c_str(), m, n, nnz, hcoo_row_ind, hcol_ind, hval, idx_base)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            gen_matrix_coo(m, n, nnz, hcoo_row_ind, hcol_ind, hval, idx_base);
        }

        // Convert COO to CSR
        hcsr_row_ptr.resize(m + 1, 0);
        for(int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptr[hcoo_row_ind[i] + 1 - idx_base];
        }

        hcsr_row_ptr[0] = idx_base;
        for(int i = 0; i < m; ++i)
        {
            hcsr_row_ptr[i + 1] += hcsr_row_ptr[i];
        }
    }

    // Allocate memory on the device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcoo_row_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcoo_row_ind = (int*)dcoo_row_ind_managed.get();

    if(!dcsr_row_ptr || !dcoo_row_ind)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dcsr_row_ptr || !dcoo_row_ind");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(
            hipsparseXcsr2coo(handle, dcsr_row_ptr, nnz, m, dcoo_row_ind, idx_base));

        // Copy output from device to host
        hcoo_row_ind.resize(nnz);
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

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            hipsparseXcsr2coo(handle, dcsr_row_ptr, nnz, m, dcoo_row_ind, idx_base);
        }

        double gpu_time_used = get_time_us();

        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            hipsparseXcsr2coo(handle, dcsr_row_ptr, nnz, m, dcoo_row_ind, idx_base);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        double bandwidth = sizeof(int) * (nnz + m + 1) / gpu_time_used / 1e6;

        printf("m\t\tn\t\tnnz\t\tGB/s\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\t%0.2lf\n", m, n, nnz, bandwidth, gpu_time_used);
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSR2COO_HPP
