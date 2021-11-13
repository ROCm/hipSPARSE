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
#ifndef TESTING_PRUNE_CSR2CSR_HPP
#define TESTING_PRUNE_CSR2CSR_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse/hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_prune_csr2csr_bad_arg(void)
{
    size_t safe_size = 100;

    int    M                      = 10;
    int    N                      = 10;
    int    nnz_A                  = 10;
    T      threshold              = static_cast<T>(1);
    int    nnz_total_dev_host_ptr = 100;
    size_t buffer_size            = 100;

    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr_A(new descr_struct);
    hipsparseMatDescr_t           descr_A = unique_ptr_descr_A->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_C(new descr_struct);
    hipsparseMatDescr_t           descr_C = unique_ptr_descr_C->descr;

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
    auto temp_buffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* csr_row_ptr_A = (int*)csr_row_ptr_A_managed.get();
    int* csr_col_ind_A = (int*)csr_col_ind_A_managed.get();
    T*   csr_val_A     = (T*)csr_val_A_managed.get();
    int* csr_row_ptr_C = (int*)csr_row_ptr_C_managed.get();
    int* csr_col_ind_C = (int*)csr_col_ind_C_managed.get();
    T*   csr_val_C     = (T*)csr_val_C_managed.get();
    T*   temp_buffer   = (T*)temp_buffer_managed.get();

    if(!csr_row_ptr_A || !csr_col_ind_A || !csr_val_A || !csr_row_ptr_C || !csr_col_ind_C
       || !csr_val_C || !temp_buffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

#if(!defined(CUDART_VERSION))
    // Test hipsparseXpruneCsr2csr_bufferSize
    status = hipsparseXpruneCsr2csr_bufferSize(nullptr,
                                               M,
                                               N,
                                               nnz_A,
                                               descr_A,
                                               csr_val_A,
                                               csr_row_ptr_A,
                                               csr_col_ind_A,
                                               &threshold,
                                               descr_C,
                                               csr_val_C,
                                               csr_row_ptr_C,
                                               csr_col_ind_C,
                                               &buffer_size);
    verify_hipsparse_status_invalid_handle(status);

    status = hipsparseXpruneCsr2csr_bufferSize(handle,
                                               M,
                                               N,
                                               nnz_A,
                                               descr_A,
                                               csr_val_A,
                                               csr_row_ptr_A,
                                               csr_col_ind_A,
                                               &threshold,
                                               descr_C,
                                               csr_val_C,
                                               csr_row_ptr_C,
                                               csr_col_ind_C,
                                               nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: buffer size is nullptr");

    // Test hipsparseXpruneCsr2csrNnz
    status = hipsparseXpruneCsr2csrNnz(nullptr,
                                       M,
                                       N,
                                       nnz_A,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       &threshold,
                                       descr_C,
                                       csr_row_ptr_C,
                                       &nnz_total_dev_host_ptr,
                                       temp_buffer);
    verify_hipsparse_status_invalid_handle(status);

    status = hipsparseXpruneCsr2csrNnz(handle,
                                       -1,
                                       N,
                                       nnz_A,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       &threshold,
                                       descr_C,
                                       csr_row_ptr_C,
                                       &nnz_total_dev_host_ptr,
                                       temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: M is invalid");

    status = hipsparseXpruneCsr2csrNnz(handle,
                                       M,
                                       -1,
                                       nnz_A,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       &threshold,
                                       descr_C,
                                       csr_row_ptr_C,
                                       &nnz_total_dev_host_ptr,
                                       temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: N is invalid");

    status = hipsparseXpruneCsr2csrNnz(handle,
                                       M,
                                       N,
                                       -1,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       &threshold,
                                       descr_C,
                                       csr_row_ptr_C,
                                       &nnz_total_dev_host_ptr,
                                       temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nnz_A is invalid");

    status = hipsparseXpruneCsr2csrNnz(handle,
                                       M,
                                       N,
                                       nnz_A,
                                       nullptr,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       &threshold,
                                       descr_C,
                                       csr_row_ptr_C,
                                       &nnz_total_dev_host_ptr,
                                       temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");

    status = hipsparseXpruneCsr2csrNnz(handle,
                                       M,
                                       N,
                                       nnz_A,
                                       descr_A,
                                       (const T*)nullptr,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       &threshold,
                                       descr_C,
                                       csr_row_ptr_C,
                                       &nnz_total_dev_host_ptr,
                                       temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_val_A is nullptr");

    status = hipsparseXpruneCsr2csrNnz(handle,
                                       M,
                                       N,
                                       nnz_A,
                                       descr_A,
                                       csr_val_A,
                                       nullptr,
                                       csr_col_ind_A,
                                       &threshold,
                                       descr_C,
                                       csr_row_ptr_C,
                                       &nnz_total_dev_host_ptr,
                                       temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr_A is nullptr");

    status = hipsparseXpruneCsr2csrNnz(handle,
                                       M,
                                       N,
                                       nnz_A,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       nullptr,
                                       &threshold,
                                       descr_C,
                                       csr_row_ptr_C,
                                       &nnz_total_dev_host_ptr,
                                       temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");

    status = hipsparseXpruneCsr2csrNnz(handle,
                                       M,
                                       N,
                                       nnz_A,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       (const T*)nullptr,
                                       descr_C,
                                       csr_row_ptr_C,
                                       &nnz_total_dev_host_ptr,
                                       temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: threshold is nullptr");

    status = hipsparseXpruneCsr2csrNnz(handle,
                                       M,
                                       N,
                                       nnz_A,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       &threshold,
                                       nullptr,
                                       csr_row_ptr_C,
                                       &nnz_total_dev_host_ptr,
                                       temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");

    status = hipsparseXpruneCsr2csrNnz(handle,
                                       M,
                                       N,
                                       nnz_A,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       &threshold,
                                       descr_C,
                                       nullptr,
                                       &nnz_total_dev_host_ptr,
                                       temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr_C is nullptr");

    status = hipsparseXpruneCsr2csrNnz(handle,
                                       M,
                                       N,
                                       nnz_A,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       &threshold,
                                       descr_C,
                                       csr_row_ptr_C,
                                       nullptr,
                                       temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: nnz_total_dev_host_ptr is nullptr");

    status = hipsparseXpruneCsr2csrNnz(handle,
                                       M,
                                       N,
                                       nnz_A,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       &threshold,
                                       descr_C,
                                       csr_row_ptr_C,
                                       &nnz_total_dev_host_ptr,
                                       nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: buffer is nullptr");

    // Test hipsparseXpruneCsr2csr
    status = hipsparseXpruneCsr2csr(nullptr,
                                    M,
                                    N,
                                    nnz_A,
                                    descr_A,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    &threshold,
                                    descr_C,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    temp_buffer);
    verify_hipsparse_status_invalid_handle(status);

    status = hipsparseXpruneCsr2csr(handle,
                                    -1,
                                    N,
                                    nnz_A,
                                    descr_A,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    &threshold,
                                    descr_C,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: M is invalid");

    status = hipsparseXpruneCsr2csr(handle,
                                    M,
                                    -1,
                                    nnz_A,
                                    descr_A,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    &threshold,
                                    descr_C,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: N is invalid");

    status = hipsparseXpruneCsr2csr(handle,
                                    M,
                                    N,
                                    -1,
                                    descr_A,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    &threshold,
                                    descr_C,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nnz_A is invalid");

    status = hipsparseXpruneCsr2csr(handle,
                                    M,
                                    N,
                                    nnz_A,
                                    nullptr,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    &threshold,
                                    descr_C,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");

    status = hipsparseXpruneCsr2csr(handle,
                                    M,
                                    N,
                                    nnz_A,
                                    descr_A,
                                    (const T*)nullptr,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    &threshold,
                                    descr_C,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_val_A is nullptr");

    status = hipsparseXpruneCsr2csr(handle,
                                    M,
                                    N,
                                    nnz_A,
                                    descr_A,
                                    csr_val_A,
                                    nullptr,
                                    csr_col_ind_A,
                                    &threshold,
                                    descr_C,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr_A is nullptr");

    status = hipsparseXpruneCsr2csr(handle,
                                    M,
                                    N,
                                    nnz_A,
                                    descr_A,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    nullptr,
                                    &threshold,
                                    descr_C,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_col_ind_A is nullptr");

    status = hipsparseXpruneCsr2csr(handle,
                                    M,
                                    N,
                                    nnz_A,
                                    descr_A,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    (const T*)nullptr,
                                    descr_C,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: threshold is nullptr");

    status = hipsparseXpruneCsr2csr(handle,
                                    M,
                                    N,
                                    nnz_A,
                                    descr_A,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    &threshold,
                                    nullptr,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");

    status = hipsparseXpruneCsr2csr(handle,
                                    M,
                                    N,
                                    nnz_A,
                                    descr_A,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    &threshold,
                                    descr_C,
                                    (T*)nullptr,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_val_C is nullptr");

    status = hipsparseXpruneCsr2csr(handle,
                                    M,
                                    N,
                                    nnz_A,
                                    descr_A,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    &threshold,
                                    descr_C,
                                    csr_val_C,
                                    nullptr,
                                    csr_col_ind_C,
                                    temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr_C is nullptr");

    status = hipsparseXpruneCsr2csr(handle,
                                    M,
                                    N,
                                    nnz_A,
                                    descr_A,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    &threshold,
                                    descr_C,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    nullptr,
                                    temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_col_ind_C is nullptr");

    status = hipsparseXpruneCsr2csr(handle,
                                    M,
                                    N,
                                    nnz_A,
                                    descr_A,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    &threshold,
                                    descr_C,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: buffer is nullptr");
#endif
}

template <typename T>
hipsparseStatus_t testing_prune_csr2csr(Arguments argus)
{
    int                  M              = argus.M;
    int                  N              = argus.N;
    T                    threshold      = static_cast<T>(argus.threshold);
    hipsparseIndexBase_t csr_idx_base_A = argus.idx_base;
    hipsparseIndexBase_t csr_idx_base_C = argus.idx_base2;
    std::string          binfile        = "";
    std::string          filename       = "";
    hipsparseStatus_t    status;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    int safe_size = 100;
    if(M == -99 && N == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        M = N = safe_size;
    }

    if(argus.timing == 1)
    {
        filename = argus.filename;
    }

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr_A(new descr_struct);
    hipsparseMatDescr_t           descr_A = unique_ptr_descr_A->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_C(new descr_struct);
    hipsparseMatDescr_t           descr_C = unique_ptr_descr_C->descr;

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_A, csr_idx_base_A));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_C, csr_idx_base_C));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        size_t safe_size = 100;

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
        auto temp_buffer_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* csr_row_ptr_A = (int*)csr_row_ptr_A_managed.get();
        int* csr_col_ind_A = (int*)csr_col_ind_A_managed.get();
        T*   csr_val_A     = (T*)csr_val_A_managed.get();
        int* csr_row_ptr_C = (int*)csr_row_ptr_C_managed.get();
        int* csr_col_ind_C = (int*)csr_col_ind_C_managed.get();
        T*   csr_val_C     = (T*)csr_val_C_managed.get();
        T*   temp_buffer   = (T*)temp_buffer_managed.get();

        if(!csr_row_ptr_A || !csr_col_ind_A || !csr_val_A || !csr_row_ptr_C || !csr_col_ind_C
           || !csr_val_C || !temp_buffer)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        status = hipsparseXpruneCsr2csr(handle,
                                        M,
                                        N,
                                        safe_size,
                                        descr_A,
                                        csr_val_A,
                                        csr_row_ptr_A,
                                        csr_col_ind_A,
                                        &threshold,
                                        descr_C,
                                        csr_val_C,
                                        csr_row_ptr_C,
                                        csr_col_ind_C,
                                        temp_buffer);

        if(M < 0 || N < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || n < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && n >= 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Read or construct CSR matrix
    std::vector<int> h_nnz_total_dev_host_ptr(1);
    std::vector<int> h_csr_row_ptr_A;
    std::vector<int> h_csr_col_ind_A;
    std::vector<T>   h_csr_val_A;
    int              nnz_A;
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(binfile.c_str(),
                           M,
                           N,
                           nnz_A,
                           h_csr_row_ptr_A,
                           h_csr_col_ind_A,
                           h_csr_val_A,
                           csr_idx_base_A)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }
    else if(argus.laplacian)
    {
        M = N = gen_2d_laplacian(
            argus.laplacian, h_csr_row_ptr_A, h_csr_col_ind_A, h_csr_val_A, csr_idx_base_A);
        nnz_A = h_csr_row_ptr_A[M];
    }
    else
    {
        std::vector<int> coo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(filename.c_str(),
                               M,
                               N,
                               nnz_A,
                               coo_row_ind,
                               h_csr_col_ind_A,
                               h_csr_val_A,
                               csr_idx_base_A)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            double scale = 0.02;
            if(M > 1000 || N > 1000)
            {
                scale = 2.0 / std::max(M, N);
            }
            nnz_A = M * scale * N;
            gen_matrix_coo(M, N, nnz_A, coo_row_ind, h_csr_col_ind_A, h_csr_val_A, csr_idx_base_A);
        }

        // Convert COO to CSR
        h_csr_row_ptr_A.resize(M + 1, 0);
        for(int i = 0; i < nnz_A; ++i)
        {
            ++h_csr_row_ptr_A[coo_row_ind[i] + 1 - csr_idx_base_A];
        }

        h_csr_row_ptr_A[0] = csr_idx_base_A;
        for(int i = 0; i < M; ++i)
        {
            h_csr_row_ptr_A[i + 1] += h_csr_row_ptr_A[i];
        }
    }

    // Allocate device memory
    auto d_nnz_total_dev_host_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
    auto d_csr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};
    auto d_csr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};
    auto d_csr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_A), device_free};
    auto d_csr_val_A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};

    int* d_nnz_total_dev_host_ptr = (int*)d_nnz_total_dev_host_ptr_managed.get();
    int* d_csr_row_ptr_C          = (int*)d_csr_row_ptr_C_managed.get();
    int* d_csr_row_ptr_A          = (int*)d_csr_row_ptr_A_managed.get();
    int* d_csr_col_ind_A          = (int*)d_csr_col_ind_A_managed.get();
    T*   d_csr_val_A              = (T*)d_csr_val_A_managed.get();

    if(!d_nnz_total_dev_host_ptr || !d_csr_row_ptr_C || !d_csr_row_ptr_A || !d_csr_col_ind_A
       || !d_csr_val_A)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!d_nnz_total_dev_host_ptr || !d_csr_row_ptr_C || "
                                        "!d_csr_row_ptr_A || !d_csr_col_ind_A || !d_csr_val_A");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Transfer.
    CHECK_HIP_ERROR(hipMemcpy(
        d_csr_row_ptr_A, h_csr_row_ptr_A.data(), sizeof(int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        d_csr_col_ind_A, h_csr_col_ind_A.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(d_csr_val_A, h_csr_val_A.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));

    size_t buffer_size = 4;
    CHECK_HIPSPARSE_ERROR(hipsparseXpruneCsr2csr_bufferSize(handle,
                                                            M,
                                                            N,
                                                            nnz_A,
                                                            descr_A,
                                                            d_csr_val_A,
                                                            d_csr_row_ptr_A,
                                                            d_csr_col_ind_A,
                                                            &threshold,
                                                            descr_C,
                                                            (const T*)nullptr,
                                                            d_csr_row_ptr_C,
                                                            (const int*)nullptr,
                                                            &buffer_size));

    auto d_temp_buffer_managed = hipsparse_unique_ptr{device_malloc(buffer_size), device_free};

    T* d_temp_buffer = (T*)d_temp_buffer_managed.get();

    if(!d_temp_buffer)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED, "!d_temp_buffer");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    auto d_threshold_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    T* d_threshold = (T*)d_threshold_managed.get();

    if(!d_threshold)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED, "!d_threshold");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    CHECK_HIP_ERROR(hipMemcpy(d_threshold, &threshold, sizeof(T), hipMemcpyHostToDevice));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXpruneCsr2csrNnz(handle,
                                                    M,
                                                    N,
                                                    nnz_A,
                                                    descr_A,
                                                    d_csr_val_A,
                                                    d_csr_row_ptr_A,
                                                    d_csr_col_ind_A,
                                                    &threshold,
                                                    descr_C,
                                                    d_csr_row_ptr_C,
                                                    &h_nnz_total_dev_host_ptr[0],
                                                    d_temp_buffer));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(hipsparseXpruneCsr2csrNnz(handle,
                                                    M,
                                                    N,
                                                    nnz_A,
                                                    descr_A,
                                                    d_csr_val_A,
                                                    d_csr_row_ptr_A,
                                                    d_csr_col_ind_A,
                                                    d_threshold,
                                                    descr_C,
                                                    d_csr_row_ptr_C,
                                                    d_nnz_total_dev_host_ptr,
                                                    d_temp_buffer));

    if(argus.unit_check)
    {
        std::vector<int> h_nnz_total_copied_from_device(1);
        CHECK_HIP_ERROR(hipMemcpy(h_nnz_total_copied_from_device.data(),
                                  d_nnz_total_dev_host_ptr,
                                  sizeof(int),
                                  hipMemcpyDeviceToHost));

        unit_check_general<int>(
            1, 1, 1, h_nnz_total_dev_host_ptr.data(), h_nnz_total_copied_from_device.data());

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        if(h_nnz_total_dev_host_ptr[0] > 0)
        {
            auto d_csr_col_ind_C_managed = hipsparse_unique_ptr{
                device_malloc(sizeof(int) * h_nnz_total_dev_host_ptr[0]), device_free};
            auto d_csr_val_C_managed = hipsparse_unique_ptr{
                device_malloc(sizeof(T) * h_nnz_total_dev_host_ptr[0]), device_free};

            int* d_csr_col_ind_C = (int*)d_csr_col_ind_C_managed.get();
            T*   d_csr_val_C     = (T*)d_csr_val_C_managed.get();

            if(!d_csr_col_ind_C || !d_csr_val_C)
            {
                verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                                "!d_csr_col_ind_C || !d_csr_val_C");
                return HIPSPARSE_STATUS_ALLOC_FAILED;
            }

            CHECK_HIPSPARSE_ERROR(hipsparseXpruneCsr2csr(handle,
                                                         M,
                                                         N,
                                                         nnz_A,
                                                         descr_A,
                                                         d_csr_val_A,
                                                         d_csr_row_ptr_A,
                                                         d_csr_col_ind_A,
                                                         &threshold,
                                                         descr_C,
                                                         d_csr_val_C,
                                                         d_csr_row_ptr_C,
                                                         d_csr_col_ind_C,
                                                         d_temp_buffer));

            std::vector<int> h_csr_row_ptr_C(M + 1);
            std::vector<int> h_csr_col_ind_C(h_nnz_total_dev_host_ptr[0]);
            std::vector<T>   h_csr_val_C(h_nnz_total_dev_host_ptr[0]);

            CHECK_HIP_ERROR(hipMemcpy(h_csr_row_ptr_C.data(),
                                      d_csr_row_ptr_C,
                                      sizeof(int) * (M + 1),
                                      hipMemcpyDeviceToHost));

            CHECK_HIP_ERROR(hipMemcpy(h_csr_col_ind_C.data(),
                                      d_csr_col_ind_C,
                                      sizeof(int) * h_nnz_total_dev_host_ptr[0],
                                      hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(h_csr_val_C.data(),
                                      d_csr_val_C,
                                      sizeof(T) * h_nnz_total_dev_host_ptr[0],
                                      hipMemcpyDeviceToHost));

            // call host and check results
            std::vector<int> h_nnz_C_cpu(1);
            std::vector<int> h_csr_row_ptr_cpu;
            std::vector<int> h_csr_col_ind_cpu;
            std::vector<T>   h_csr_val_cpu;

            host_prune_csr_to_csr(M,
                                  N,
                                  nnz_A,
                                  h_csr_row_ptr_A,
                                  h_csr_col_ind_A,
                                  h_csr_val_A,
                                  h_nnz_C_cpu[0],
                                  h_csr_row_ptr_cpu,
                                  h_csr_col_ind_cpu,
                                  h_csr_val_cpu,
                                  csr_idx_base_A,
                                  csr_idx_base_C,
                                  threshold);

            unit_check_general<int>(1, 1, 1, h_nnz_C_cpu.data(), h_nnz_total_dev_host_ptr.data());
            unit_check_general<int>(
                1, (M + 1), 1, h_csr_row_ptr_cpu.data(), h_csr_row_ptr_C.data());
            unit_check_general<int>(1,
                                    h_nnz_total_dev_host_ptr[0],
                                    1,
                                    h_csr_col_ind_cpu.data(),
                                    h_csr_col_ind_C.data());
            unit_check_general<T>(
                1, h_nnz_total_dev_host_ptr[0], 1, h_csr_val_cpu.data(), h_csr_val_C.data());
        }
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_PRUNE_CSR2CSR_HPP
