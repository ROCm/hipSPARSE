/* ************************************************************************
 * Copyright (C) 2020 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_CSR2BSR_HPP
#define TESTING_CSR2BSR_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_csr2bsr_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                  m            = 100;
    int                  n            = 100;
    int                  safe_size    = 100;
    int                  block_dim    = 2;
    hipsparseIndexBase_t csr_idx_base = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t bsr_idx_base = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseDirection_t dir          = HIPSPARSE_DIRECTION_ROW;
    hipsparseStatus_t    status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t            csr_descr = unique_ptr_csr_descr->descr;
    std::unique_ptr<descr_struct>  unique_ptr_bsr_descr(new descr_struct);
    hipsparseMatDescr_t            bsr_descr = unique_ptr_bsr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, csr_idx_base);
    hipsparseSetMatIndexBase(bsr_descr, bsr_idx_base);

    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto bsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* csr_col_ind = (int*)csr_col_ind_managed.get();
    T*   csr_val     = (T*)csr_val_managed.get();
    int* bsr_row_ptr = (int*)bsr_row_ptr_managed.get();
    int* bsr_col_ind = (int*)bsr_col_ind_managed.get();
    T*   bsr_val     = (T*)bsr_val_managed.get();

    if(!csr_row_ptr || !csr_col_ind || !csr_val || !bsr_row_ptr || !bsr_col_ind || !bsr_val)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing hipsparseXcsr2bsrNnz()

    int bsr_nnzb;

    // Test invalid handle
    status = hipsparseXcsr2bsrNnz(nullptr,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  bsr_row_ptr,
                                  &bsr_nnzb);
    verify_hipsparse_status_invalid_handle(status);

    // Test invalid pointers
    status = hipsparseXcsr2bsrNnz(handle,
                                  dir,
                                  m,
                                  n,
                                  nullptr,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  bsr_row_ptr,
                                  &bsr_nnzb);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_descr is nullptr");

    status = hipsparseXcsr2bsrNnz(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  nullptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  bsr_row_ptr,
                                  &bsr_nnzb);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");

    status = hipsparseXcsr2bsrNnz(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  nullptr,
                                  bsr_row_ptr,
                                  &bsr_nnzb);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_descr is nullptr");

    status = hipsparseXcsr2bsrNnz(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  nullptr,
                                  &bsr_nnzb);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr is nullptr");

    status = hipsparseXcsr2bsrNnz(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  bsr_row_ptr,
                                  nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_nnzb is nullptr");

    // Test invalid sizes
    status = hipsparseXcsr2bsrNnz(handle,
                                  dir,
                                  -1,
                                  n,
                                  csr_descr,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  bsr_row_ptr,
                                  &bsr_nnzb);
    verify_hipsparse_status_invalid_size(status, "Error: m is invalid");

    status = hipsparseXcsr2bsrNnz(handle,
                                  dir,
                                  m,
                                  -1,
                                  csr_descr,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  bsr_row_ptr,
                                  &bsr_nnzb);
    verify_hipsparse_status_invalid_size(status, "Error: n is invalid");

    status = hipsparseXcsr2bsrNnz(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  -1,
                                  bsr_descr,
                                  bsr_row_ptr,
                                  &bsr_nnzb);
    verify_hipsparse_status_invalid_size(status, "Error: block_dim is invalid");

    // Testing hipsparseXcsr2bsr()

    // Test invalid handle
    status = hipsparseXcsr2bsr(nullptr,
                               dir,
                               m,
                               n,
                               csr_descr,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind,
                               block_dim,
                               bsr_descr,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind);
    verify_hipsparse_status_invalid_handle(status);

    // Test invalid pointers
    status = hipsparseXcsr2bsr(handle,
                               dir,
                               m,
                               n,
                               nullptr,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind,
                               block_dim,
                               bsr_descr,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_descr is nullptr");

    status = hipsparseXcsr2bsr(handle,
                               dir,
                               m,
                               n,
                               csr_descr,
                               (T*)nullptr,
                               csr_row_ptr,
                               csr_col_ind,
                               block_dim,
                               bsr_descr,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");

    status = hipsparseXcsr2bsr(handle,
                               dir,
                               m,
                               n,
                               csr_descr,
                               csr_val,
                               nullptr,
                               csr_col_ind,
                               block_dim,
                               bsr_descr,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");

    status = hipsparseXcsr2bsr(handle,
                               dir,
                               m,
                               n,
                               csr_descr,
                               csr_val,
                               csr_row_ptr,
                               nullptr,
                               block_dim,
                               bsr_descr,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");

    status = hipsparseXcsr2bsr(handle,
                               dir,
                               m,
                               n,
                               csr_descr,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind,
                               block_dim,
                               nullptr,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_descr is nullptr");

    status = hipsparseXcsr2bsr(handle,
                               dir,
                               m,
                               n,
                               csr_descr,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind,
                               block_dim,
                               bsr_descr,
                               (T*)nullptr,
                               bsr_row_ptr,
                               bsr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_val is nullptr");

    status = hipsparseXcsr2bsr(handle,
                               dir,
                               m,
                               n,
                               csr_descr,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind,
                               block_dim,
                               bsr_descr,
                               bsr_val,
                               nullptr,
                               bsr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr is nullptr");

    status = hipsparseXcsr2bsr(handle,
                               dir,
                               m,
                               n,
                               csr_descr,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind,
                               block_dim,
                               bsr_descr,
                               bsr_val,
                               bsr_row_ptr,
                               nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind is nullptr");

    // Test invalid sizes
    status = hipsparseXcsr2bsr(handle,
                               dir,
                               -1,
                               n,
                               csr_descr,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind,
                               block_dim,
                               bsr_descr,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind);
    verify_hipsparse_status_invalid_size(status, "Error: m is invalid");

    status = hipsparseXcsr2bsr(handle,
                               dir,
                               m,
                               -1,
                               csr_descr,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind,
                               block_dim,
                               bsr_descr,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind);
    verify_hipsparse_status_invalid_size(status, "Error: n is invalid");

    status = hipsparseXcsr2bsr(handle,
                               dir,
                               m,
                               n,
                               csr_descr,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind,
                               -1,
                               bsr_descr,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind);
    verify_hipsparse_status_invalid_size(status, "Error: block_dim is invalid");
#endif
}

template <typename T>
hipsparseStatus_t testing_csr2bsr(Arguments argus)
{
    int                  m            = argus.M;
    int                  n            = argus.N;
    int                  block_dim    = argus.block_dim;
    hipsparseIndexBase_t csr_idx_base = argus.idx_base;
    hipsparseIndexBase_t bsr_idx_base = argus.idx_base2;
    hipsparseDirection_t dir          = argus.dirA;
    std::string          binfile      = "";
    std::string          filename     = "";
    hipsparseStatus_t    status;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    int safe_size = std::max(100, std::max(m, n));
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
    std::unique_ptr<descr_struct>  unique_ptr_bsr_descr(new descr_struct);
    hipsparseMatDescr_t            bsr_descr = unique_ptr_bsr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, csr_idx_base);
    hipsparseSetMatIndexBase(bsr_descr, bsr_idx_base);

    if(block_dim == 1)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        // cusparse does not support asynchronous execution for block_dim == 1
        return HIPSPARSE_STATUS_SUCCESS;
#endif
    }

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || block_dim <= 0)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        // Do not test args in cusparse
        return HIPSPARSE_STATUS_SUCCESS;
#endif
        auto dcsr_row_ptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
        auto dcsr_col_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dcsr_val_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dbsr_row_ptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
        auto dbsr_col_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dbsr_val_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
        int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
        T*   dcsr_val     = (T*)dcsr_val_managed.get();
        int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();
        int* dbsr_col_ind = (int*)dbsr_col_ind_managed.get();
        T*   dbsr_val     = (T*)dbsr_val_managed.get();

        // row pointer need to be valid
        CHECK_HIP_ERROR(hipMemset(dcsr_row_ptr, 0, sizeof(int) * (safe_size + 1)));

        if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dbsr_row_ptr || !dbsr_col_ind
           || !dbsr_val)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || "
                                            "!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || ");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        int bsr_nnzb;

        status = hipsparseXcsr2bsrNnz(handle,
                                      dir,
                                      m,
                                      n,
                                      csr_descr,
                                      dcsr_row_ptr,
                                      dcsr_col_ind,
                                      block_dim,
                                      bsr_descr,
                                      dbsr_row_ptr,
                                      &bsr_nnzb);

        if(m < 0 || n < 0 || block_dim <= 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || block_dim <= 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && n >= 0 && block_dim > 0");
        }

        status = hipsparseXcsr2bsr(handle,
                                   dir,
                                   m,
                                   n,
                                   csr_descr,
                                   dcsr_val,
                                   dcsr_row_ptr,
                                   dcsr_col_ind,
                                   block_dim,
                                   bsr_descr,
                                   dbsr_val,
                                   dbsr_row_ptr,
                                   dbsr_col_ind);

        if(m < 0 || n < 0 || block_dim <= 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || block_dim <= 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && n >= 0 && block_dim > 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Read or construct CSR matrix
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;
    int              nnz;
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(
               binfile.c_str(), m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, csr_idx_base)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }
    else if(argus.laplacian)
    {
        m = n
            = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcsr_col_ind, hcsr_val, csr_idx_base);
        nnz = hcsr_row_ptr[m];
    }
    else
    {
        std::vector<int> coo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(
                   filename.c_str(), m, n, nnz, coo_row_ind, hcsr_col_ind, hcsr_val, csr_idx_base)
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
            nnz = m * scale * n;
            gen_matrix_coo(m, n, nnz, coo_row_ind, hcsr_col_ind, hcsr_val, csr_idx_base);
        }

        // Convert COO to CSR
        hcsr_row_ptr.resize(m + 1, 0);
        for(int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptr[coo_row_ind[i] + 1 - csr_idx_base];
        }

        hcsr_row_ptr[0] = csr_idx_base;
        for(int i = 0; i < m; ++i)
        {
            hcsr_row_ptr[i + 1] += hcsr_row_ptr[i];
        }
    }

    int mb = (m + block_dim - 1) / block_dim;

    // Allocate memory on the device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val     = (T*)dcsr_val_managed.get();
    int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dbsr_row_ptr)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || "
                                        "!dbsr_row_ptr");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        // Obtain BSR nnzb first on the host and then using the device and ensure they give the same results
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        int hbsr_nnzb;
        CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsrNnz(handle,
                                                   dir,
                                                   m,
                                                   n,
                                                   csr_descr,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind,
                                                   block_dim,
                                                   bsr_descr,
                                                   dbsr_row_ptr,
                                                   &hbsr_nnzb));

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));

        auto dbsr_nnzb_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
        int* dbsr_nnzb         = (int*)dbsr_nnzb_managed.get();
        CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsrNnz(handle,
                                                   dir,
                                                   m,
                                                   n,
                                                   csr_descr,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind,
                                                   block_dim,
                                                   bsr_descr,
                                                   dbsr_row_ptr,
                                                   dbsr_nnzb));

        int hbsr_nnzb_copied_from_device;
        CHECK_HIP_ERROR(hipMemcpy(
            &hbsr_nnzb_copied_from_device, dbsr_nnzb, sizeof(int), hipMemcpyDeviceToHost));

        // Check that using host and device pointer mode gives the same result
        unit_check_general(1, 1, 1, &hbsr_nnzb_copied_from_device, &hbsr_nnzb);

        // Allocate memory on the device
        auto dbsr_col_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * hbsr_nnzb), device_free};
        auto dbsr_val_managed = hipsparse_unique_ptr{
            device_malloc(sizeof(T) * hbsr_nnzb * block_dim * block_dim), device_free};

        int* dbsr_col_ind = (int*)dbsr_col_ind_managed.get();
        T*   dbsr_val     = (T*)dbsr_val_managed.get();

        if(!dbsr_col_ind || !dbsr_val)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!bsr_col_ind || !bsr_val");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsr(handle,
                                                dir,
                                                m,
                                                n,
                                                csr_descr,
                                                dcsr_val,
                                                dcsr_row_ptr,
                                                dcsr_col_ind,
                                                block_dim,
                                                bsr_descr,
                                                dbsr_val,
                                                dbsr_row_ptr,
                                                dbsr_col_ind));

        // Copy output from device to host
        std::vector<int> hbsr_row_ptr(mb + 1);
        std::vector<int> hbsr_col_ind(hbsr_nnzb);
        std::vector<T>   hbsr_val(hbsr_nnzb * block_dim * block_dim);

        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_row_ptr.data(), dbsr_row_ptr, sizeof(int) * (mb + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_col_ind.data(), dbsr_col_ind, sizeof(int) * hbsr_nnzb, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val.data(),
                                  dbsr_val,
                                  sizeof(T) * hbsr_nnzb * block_dim * block_dim,
                                  hipMemcpyDeviceToHost));

        // Host csr2bsr conversion
        std::vector<int> hbsr_row_ptr_gold(mb + 1);
        std::vector<int> hbsr_col_ind_gold(hbsr_nnzb, 0);
        std::vector<T>   hbsr_val_gold(hbsr_nnzb * block_dim * block_dim);

        // call host csr2bsr here
        int bsr_nnzb_gold;
        host_csr_to_bsr<T>(dir,
                           m,
                           n,
                           block_dim,
                           bsr_nnzb_gold,
                           csr_idx_base,
                           hcsr_row_ptr,
                           hcsr_col_ind,
                           hcsr_val,
                           bsr_idx_base,
                           hbsr_row_ptr_gold,
                           hbsr_col_ind_gold,
                           hbsr_val_gold);

        // Unit check
        unit_check_general(1, 1, 1, &bsr_nnzb_gold, &hbsr_nnzb);
        unit_check_general(1, mb + 1, 1, hbsr_row_ptr_gold.data(), hbsr_row_ptr.data());
        unit_check_general(1, hbsr_nnzb, 1, hbsr_col_ind_gold.data(), hbsr_col_ind.data());
        unit_check_general(
            1, hbsr_nnzb * block_dim * block_dim, 1, hbsr_val_gold.data(), hbsr_val.data());
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSR2BSR_HPP
