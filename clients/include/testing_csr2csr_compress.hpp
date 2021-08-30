/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef TESTING_CSR2CSR_COMPRESS_HPP
#define TESTING_CSR2CSR_COMPRESS_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse.h>
#include <iostream>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_csr2csr_compress_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVIDIA__
    // do not test for bad args
    return;
#endif
    int                  m            = 100;
    int                  n            = 100;
    int                  nnz_A        = 100;
    int                  safe_size    = 100;
    T                    tol          = make_DataType<T>(0);
    hipsparseIndexBase_t csr_idx_base = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseStatus_t    status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t            csr_descr = unique_ptr_csr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, csr_idx_base);

    auto csr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto csr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_col_ind_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto nnz_per_row_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto nnz_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};

    int* csr_row_ptr_A = (int*)csr_row_ptr_A_managed.get();
    int* csr_col_ind_A = (int*)csr_col_ind_A_managed.get();
    T*   csr_val_A     = (T*)csr_val_A_managed.get();
    int* csr_row_ptr_C = (int*)csr_row_ptr_C_managed.get();
    int* csr_col_ind_C = (int*)csr_col_ind_C_managed.get();
    T*   csr_val_C     = (T*)csr_val_C_managed.get();
    int* nnz_per_row   = (int*)nnz_per_row_managed.get();
    int* nnz_C         = (int*)nnz_C_managed.get();

    if(!csr_row_ptr_A || !csr_col_ind_A || !csr_val_A || !csr_row_ptr_C || !csr_col_ind_C
       || !csr_val_C || !nnz_per_row || !nnz_C)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing hipsparseXnnz_compress()

    // Test invalid handle
    status = hipsparseXnnz_compress(
        nullptr, m, csr_descr, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol);
    verify_hipsparse_status_invalid_handle(status);

    // Test invalid pointers
    status = hipsparseXnnz_compress(
        handle, m, nullptr, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol);
    verify_hipsparse_status_invalid_pointer(status, "Error: Matrix descriptor is invalid");

    status
        = hipsparseXnnz_compress(handle, m, csr_descr, csr_val_A, nullptr, nnz_per_row, nnz_C, tol);
    verify_hipsparse_status_invalid_pointer(status, "Error: CSR row pointer array is invalid");

    status = hipsparseXnnz_compress(
        handle, m, csr_descr, csr_val_A, csr_row_ptr_A, nullptr, nnz_C, tol);
    verify_hipsparse_status_invalid_pointer(status,
                                            "Error: Number of elements per row array is invalid");

    status = hipsparseXnnz_compress(
        handle, m, csr_descr, csr_val_A, csr_row_ptr_A, nnz_per_row, nullptr, tol);
    verify_hipsparse_status_invalid_pointer(status,
                                            "Error: Total number of elements pointer is invalid");

    // Test invalid size
    status = hipsparseXnnz_compress(
        handle, -1, csr_descr, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol);
    verify_hipsparse_status_invalid_size(status, "Error: Matrix size is invalid");

    // Test invalid tolerance
    status = hipsparseXnnz_compress(
        handle, m, csr_descr, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, make_DataType<T>(-1));
    verify_hipsparse_status_invalid_size(status, "Error: Tolerance is invalid");

    // Testing hipsparseXcsr2csr_compress()

    // Test invalid handle
    status = hipsparseXcsr2csr_compress(nullptr,
                                        m,
                                        n,
                                        csr_descr,
                                        csr_val_A,
                                        csr_col_ind_A,
                                        csr_row_ptr_A,
                                        nnz_A,
                                        nnz_per_row,
                                        csr_val_C,
                                        csr_col_ind_C,
                                        csr_row_ptr_C,
                                        tol);
    verify_hipsparse_status_invalid_handle(status);

    // Test invalid pointers
    status = hipsparseXcsr2csr_compress(handle,
                                        m,
                                        n,
                                        nullptr,
                                        csr_val_A,
                                        csr_col_ind_A,
                                        csr_row_ptr_A,
                                        nnz_A,
                                        nnz_per_row,
                                        csr_val_C,
                                        csr_col_ind_C,
                                        csr_row_ptr_C,
                                        tol);
    verify_hipsparse_status_invalid_pointer(status, "Error: Matrix descriptor is invalid");

    status = hipsparseXcsr2csr_compress(handle,
                                        m,
                                        n,
                                        csr_descr,
                                        (const T*)nullptr,
                                        csr_col_ind_A,
                                        csr_row_ptr_A,
                                        nnz_A,
                                        nnz_per_row,
                                        csr_val_C,
                                        csr_col_ind_C,
                                        csr_row_ptr_C,
                                        tol);
    verify_hipsparse_status_invalid_pointer(status, "Error: CSR matrix values array is invalid");

    status = hipsparseXcsr2csr_compress(handle,
                                        m,
                                        n,
                                        csr_descr,
                                        csr_val_A,
                                        nullptr,
                                        csr_row_ptr_A,
                                        nnz_A,
                                        nnz_per_row,
                                        csr_val_C,
                                        csr_col_ind_C,
                                        csr_row_ptr_C,
                                        tol);
    verify_hipsparse_status_invalid_pointer(status,
                                            "Error: CSR matrix column indices array is invalid");

    status = hipsparseXcsr2csr_compress(handle,
                                        m,
                                        n,
                                        csr_descr,
                                        csr_val_A,
                                        csr_col_ind_A,
                                        nullptr,
                                        nnz_A,
                                        nnz_per_row,
                                        csr_val_C,
                                        csr_col_ind_C,
                                        csr_row_ptr_C,
                                        tol);
    verify_hipsparse_status_invalid_pointer(status,
                                            "Error: CSR matrix row pointer array is invalid");

    status = hipsparseXcsr2csr_compress(handle,
                                        m,
                                        n,
                                        csr_descr,
                                        csr_val_A,
                                        csr_col_ind_A,
                                        csr_row_ptr_A,
                                        nnz_A,
                                        nullptr,
                                        csr_val_C,
                                        csr_col_ind_C,
                                        csr_row_ptr_C,
                                        tol);
    verify_hipsparse_status_invalid_pointer(status,
                                            "Error: Number of elements per row array is invalid");

    status = hipsparseXcsr2csr_compress(handle,
                                        m,
                                        n,
                                        csr_descr,
                                        csr_val_A,
                                        csr_col_ind_A,
                                        csr_row_ptr_A,
                                        nnz_A,
                                        nnz_per_row,
                                        (T*)nullptr,
                                        csr_col_ind_C,
                                        csr_row_ptr_C,
                                        tol);
    verify_hipsparse_status_invalid_pointer(status, "Error: CSR matrix values array is invalid");

    status = hipsparseXcsr2csr_compress(handle,
                                        m,
                                        n,
                                        csr_descr,
                                        csr_val_A,
                                        csr_col_ind_A,
                                        csr_row_ptr_A,
                                        nnz_A,
                                        nnz_per_row,
                                        csr_val_C,
                                        nullptr,
                                        csr_row_ptr_C,
                                        tol);
    verify_hipsparse_status_invalid_pointer(status,
                                            "Error: CSR matrix column indices array is invalid");

    status = hipsparseXcsr2csr_compress(handle,
                                        m,
                                        n,
                                        csr_descr,
                                        csr_val_A,
                                        csr_col_ind_A,
                                        csr_row_ptr_A,
                                        nnz_A,
                                        nnz_per_row,
                                        csr_val_C,
                                        csr_col_ind_C,
                                        nullptr,
                                        tol);
    verify_hipsparse_status_invalid_pointer(status,
                                            "Error: CSR matrix row pointer array is invalid");

    // Test invalid sizes
    status = hipsparseXcsr2csr_compress(handle,
                                        -1,
                                        n,
                                        csr_descr,
                                        csr_val_A,
                                        csr_col_ind_A,
                                        csr_row_ptr_A,
                                        nnz_A,
                                        nnz_per_row,
                                        csr_val_C,
                                        csr_col_ind_C,
                                        csr_row_ptr_C,
                                        tol);
    verify_hipsparse_status_invalid_size(status, "Error: Matrix size is invalid");

    status = hipsparseXcsr2csr_compress(handle,
                                        m,
                                        -1,
                                        csr_descr,
                                        csr_val_A,
                                        csr_col_ind_A,
                                        csr_row_ptr_A,
                                        nnz_A,
                                        nnz_per_row,
                                        csr_val_C,
                                        csr_col_ind_C,
                                        csr_row_ptr_C,
                                        tol);
    verify_hipsparse_status_invalid_size(status, "Error: Matrix size is invalid");

    status = hipsparseXcsr2csr_compress(handle,
                                        m,
                                        n,
                                        csr_descr,
                                        csr_val_A,
                                        csr_col_ind_A,
                                        csr_row_ptr_A,
                                        -1,
                                        nnz_per_row,
                                        csr_val_C,
                                        csr_col_ind_C,
                                        csr_row_ptr_C,
                                        tol);
    verify_hipsparse_status_invalid_size(status, "Error: Matrix size is invalid");

    // Test invalid tolerance
    status = hipsparseXcsr2csr_compress(handle,
                                        m,
                                        n,
                                        csr_descr,
                                        csr_val_A,
                                        csr_col_ind_A,
                                        csr_row_ptr_A,
                                        nnz_A,
                                        nnz_per_row,
                                        csr_val_C,
                                        csr_col_ind_C,
                                        csr_row_ptr_C,
                                        static_cast<T>(-1));
    verify_hipsparse_status_invalid_value(status, "Error: Tolerance is invalid");
}

template <typename T>
hipsparseStatus_t testing_csr2csr_compress(Arguments argus)
{
    int m   = argus.M;
    int n   = argus.N;
    T   tol = make_DataType<T>(argus.alpha);

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

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t            csr_descr = unique_ptr_csr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, idx_base);

    hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE);

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || testing_real(tol) < testing_real(make_DataType<T>(0)))
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        // Do not test args in cusparse
        return HIPSPARSE_STATUS_SUCCESS;
#endif
        auto dcsr_row_ptr_A_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dcsr_col_ind_A_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dcsr_val_A_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dcsr_row_ptr_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dcsr_col_ind_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dcsr_val_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dnnz_per_row_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dnnz_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};

        int* dcsr_row_ptr_A = (int*)dcsr_row_ptr_A_managed.get();
        int* dcsr_col_ind_A = (int*)dcsr_col_ind_A_managed.get();
        T*   dcsr_val_A     = (T*)dcsr_val_A_managed.get();
        int* dcsr_row_ptr_C = (int*)dcsr_row_ptr_C_managed.get();
        int* dcsr_col_ind_C = (int*)dcsr_col_ind_C_managed.get();
        T*   dcsr_val_C     = (T*)dcsr_val_C_managed.get();
        int* dnnz_per_row   = (int*)dnnz_per_row_managed.get();
        int* dnnz_C         = (int*)dnnz_C_managed.get();

        if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_C || !dcsr_col_ind_C
           || !dcsr_val_C || !dnnz_per_row || !dnnz_C)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || "
                                            "!dcsr_row_ptr_C || !dcsr_col_ind_C || !dcsr_val_C || "
                                            "!dnnz_per_row || !dnnz_C");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        status = hipsparseXcsr2csr_compress(handle,
                                            m,
                                            n,
                                            csr_descr,
                                            dcsr_val_A,
                                            dcsr_col_ind_A,
                                            dcsr_row_ptr_A,
                                            safe_size,
                                            dnnz_per_row,
                                            dcsr_val_C,
                                            dcsr_col_ind_C,
                                            dcsr_row_ptr_C,
                                            tol);

        if(m < 0 || n < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || n < 0");
        }
        else if(testing_real(tol) < testing_real(make_DataType<T>(0)))
        {
            verify_hipsparse_status_invalid_value(status, "Error: real(tol) < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && n >= 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Host CSR matrix
    std::vector<int> hcsr_row_ptr_A;
    std::vector<int> hcsr_col_ind_A;
    std::vector<T>   hcsr_val_A;

    // Sample initial COO matrix on CPU
    int hnnz_A;
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(
               binfile.c_str(), m, n, hnnz_A, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idx_base)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }
    else if(argus.laplacian)
    {
        m = n = gen_2d_laplacian(
            argus.laplacian, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idx_base);
        hnnz_A = hcsr_row_ptr_A[m];
    }
    else
    {
        std::vector<int> hcoo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(filename.c_str(),
                               m,
                               n,
                               hnnz_A,
                               hcoo_row_ind,
                               hcsr_col_ind_A,
                               hcsr_val_A,
                               idx_base)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            double scale = 0.02;
            if(m > 1000 || n > 1000)
            {
                scale = 2.0 / std::max(m, n);
            }
            hnnz_A = m * scale * n;

            gen_matrix_coo(m, n, hnnz_A, hcoo_row_ind, hcsr_col_ind_A, hcsr_val_A, idx_base);
        }

        // Convert COO to CSR
        hcsr_row_ptr_A.resize(m + 1, 0);
        for(int i = 0; i < hnnz_A; ++i)
        {
            ++hcsr_row_ptr_A[hcoo_row_ind[i] + 1 - idx_base];
        }

        hcsr_row_ptr_A[0] = idx_base;
        for(int i = 0; i < m; ++i)
        {
            hcsr_row_ptr_A[i + 1] += hcsr_row_ptr_A[i];
        }
    }

    // Allocate memory on the device
    auto dcsr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * hnnz_A), device_free};
    auto dcsr_val_A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * hnnz_A), device_free};
    auto dcsr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dnnz_per_row_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * m), device_free};

    int* dcsr_row_ptr_A = (int*)dcsr_row_ptr_A_managed.get();
    int* dcsr_col_ind_A = (int*)dcsr_col_ind_A_managed.get();
    T*   dcsr_val_A     = (T*)dcsr_val_A_managed.get();
    int* dcsr_row_ptr_C = (int*)dcsr_row_ptr_C_managed.get();
    int* dnnz_per_row   = (int*)dnnz_per_row_managed.get();

    if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_C || !dnnz_per_row)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || "
                                        "!dcsr_row_ptr_C || !dnnz_per_row");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind_A, hcsr_col_ind_A.data(), sizeof(int) * hnnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val_A, hcsr_val_A.data(), sizeof(T) * hnnz_A, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        // Use both host and device pointers for nnz_C and confirm they give the same answer
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        int hnnz_C;
        CHECK_HIPSPARSE_ERROR(hipsparseXnnz_compress(
            handle, m, csr_descr, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, &hnnz_C, tol));

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));

        auto dnnz_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
        int* dnnz_C         = (int*)dnnz_C_managed.get();
        CHECK_HIPSPARSE_ERROR(hipsparseXnnz_compress(
            handle, m, csr_descr, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, dnnz_C, tol));

        int hnnz_C_copied_from_device;
        CHECK_HIP_ERROR(
            hipMemcpy(&hnnz_C_copied_from_device, dnnz_C, sizeof(int), hipMemcpyDeviceToHost));

        unit_check_general(1, 1, 1, &hnnz_C_copied_from_device, &hnnz_C);

        if(hnnz_C == 0)
        {
            return HIPSPARSE_STATUS_SUCCESS;
        }

        // Allocate device memory for compressed CSR columns indices and values
        auto dcsr_col_ind_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * hnnz_C), device_free};
        auto dcsr_val_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * hnnz_C), device_free};

        int* dcsr_col_ind_C = (int*)dcsr_col_ind_C_managed.get();
        T*   dcsr_val_C     = (T*)dcsr_val_C_managed.get();

        if(!dcsr_col_ind_C || !dcsr_val_C)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!dcsr_col_ind_C || !dcsr_val_C");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        CHECK_HIPSPARSE_ERROR(hipsparseXcsr2csr_compress(handle,
                                                         m,
                                                         n,
                                                         csr_descr,
                                                         dcsr_val_A,
                                                         dcsr_col_ind_A,
                                                         dcsr_row_ptr_A,
                                                         hnnz_A,
                                                         dnnz_per_row,
                                                         dcsr_val_C,
                                                         dcsr_col_ind_C,
                                                         dcsr_row_ptr_C,
                                                         tol));

        // Copy output from device to host
        std::vector<int> hcsr_row_ptr_C(m + 1);
        std::vector<int> hcsr_col_ind_C(hnnz_C);
        std::vector<T>   hcsr_val_C(hnnz_C);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr_C.data(), dcsr_row_ptr_C, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind_C.data(), dcsr_col_ind_C, sizeof(int) * hnnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C.data(), dcsr_val_C, sizeof(T) * hnnz_C, hipMemcpyDeviceToHost));

        // Host csr2csc conversion
        std::vector<int> hcsr_row_ptr_C_gold;
        std::vector<int> hcsr_col_ind_C_gold;
        std::vector<T>   hcsr_val_gold;

        // Call host conversion here
        host_csr_to_csr_compress<T>(m,
                                    n,
                                    hcsr_row_ptr_A,
                                    hcsr_col_ind_A,
                                    hcsr_val_A,
                                    hcsr_row_ptr_C_gold,
                                    hcsr_col_ind_C_gold,
                                    hcsr_val_gold,
                                    idx_base,
                                    tol);

        // Unit check
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C.data());
        unit_check_general(1, hnnz_C, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C.data());
        unit_check_general(1, hnnz_C, 1, hcsr_val_gold.data(), hcsr_val_C.data());
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSR2CSR_COMPRESS_HPP
