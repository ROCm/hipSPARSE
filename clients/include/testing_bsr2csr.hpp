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
#ifndef TESTING_BSR2CSR_HPP
#define TESTING_BSR2CSR_HPP

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
void testing_bsr2csr_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVCC__
    // do not test for bad args
    return;
#endif
    int               m          = 100;
    int               n          = 100;
    int               nnz        = 100;
    int               safe_size  = 100;
    int               block_dim  = 2;
    hipsparseIndexBase_t csr_idx_base  = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t bsr_idx_base  = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseDirection_t dir     = HIPSPARSE_DIRECTION_ROW;
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct> unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t           csr_descr = unique_ptr_csr_descr->descr;
    std::unique_ptr<descr_struct> unique_ptr_bsr_descr(new descr_struct);
    hipsparseMatDescr_t           bsr_descr = unique_ptr_bsr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, csr_idx_base);
    hipsparseSetMatIndexBase(bsr_descr, bsr_idx_base);

    auto bsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* bsr_row_ptr = (int*)bsr_row_ptr_managed.get();
    int* bsr_col_ind = (int*)bsr_col_ind_managed.get();
    T*   bsr_val     = (T*)bsr_val_managed.get();
    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* csr_col_ind = (int*)csr_col_ind_managed.get();
    T*   csr_val     = (T*)csr_val_managed.get();

    if(!bsr_row_ptr || !bsr_col_ind || !bsr_val || !csr_row_ptr || !csr_col_ind || !csr_val)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing hipsparseXbsr2csr()

    // Testing for (handle == nullptr)
    {
        status = hipsparseXbsr2csr(nullptr,
                                   dir,
                                   m,
                                   n,
                                   bsr_descr,
                                   bsr_val,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   block_dim,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: handle is nullptr");
    }

    // Testing for (bsr_descr == nullptr)
    {
        status = hipsparseXbsr2csr(handle,
                                   dir,
                                   m,
                                   n,
                                   nullptr,
                                   bsr_val,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   block_dim,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: bsr_descr is nullptr");
    }

    // Testing for (bsr_val == nullptr)
    {
        status = hipsparseXbsr2csr(handle,
                                   dir,
                                   m,
                                   n,
                                   bsr_descr,
                                   (T*)nullptr,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   block_dim,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: bsr_val is nullptr");
    }

    // Testing for (bsr_row_ptr == nullptr)
    {
        status = hipsparseXbsr2csr(handle,
                                   dir,
                                   m,
                                   n,
                                   bsr_descr,
                                   bsr_val,
                                   nullptr,
                                   bsr_col_ind,
                                   block_dim,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr is nullptr");
    }

    // Testing for (bsr_col_ind == nullptr)
    {
        status = hipsparseXbsr2csr(handle,
                                   dir,
                                   m,
                                   n,
                                   bsr_descr,
                                   bsr_val,
                                   bsr_row_ptr,
                                   nullptr,
                                   block_dim,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind is nullptr");
    }

    // Testing for (csr_descr == nullptr)
    {
        status = hipsparseXbsr2csr(handle,
                                   dir,
                                   m,
                                   n,
                                   bsr_descr,
                                   bsr_val,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   block_dim,
                                   nullptr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_descr is nullptr");
    }

    // Testing for (csr_val == nullptr)
    {
        status = hipsparseXbsr2csr(handle,
                                   dir,
                                   m,
                                   n,
                                   bsr_descr,
                                   bsr_val,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   block_dim,
                                   csr_descr,
                                   (T*)nullptr,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");
    }

    // Testing for (csr_row_ptr == nullptr)
    {
        status = hipsparseXbsr2csr(handle,
                                   dir,
                                   m,
                                   n,
                                   bsr_descr,
                                   bsr_val,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   block_dim,
                                   csr_descr,
                                   csr_val,
                                   nullptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    // Testing for (csr_col_ind == nullptr)
    {
        status = hipsparseXbsr2csr(handle,
                                   dir,
                                   m,
                                   n,
                                   bsr_descr,
                                   bsr_val,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   block_dim,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   nullptr);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }
}

template <typename T>
hipsparseStatus_t testing_bsr2csr(Arguments argus)
{
    int                  m         = argus.M;
    int                  n         = argus.N;
    int                  block_dim = argus.block_dim;
    int                  safe_size = 100;
    hipsparseIndexBase_t csr_idx_base  = argus.idx_base;
    hipsparseIndexBase_t bsr_idx_base  = argus.idx_base2;
    hipsparseDirection_t dir       = argus.dirA;
    std::string          binfile   = "";
    std::string          filename  = "";
    hipsparseStatus_t    status;
    int         mb        = (m + block_dim - 1) / block_dim;
    int         nb        = (n + block_dim - 1) / block_dim;

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
    std::unique_ptr<descr_struct> unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t           csr_descr = unique_ptr_csr_descr->descr;
    std::unique_ptr<descr_struct> unique_ptr_bsr_descr(new descr_struct);
    hipsparseMatDescr_t           bsr_descr = unique_ptr_bsr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, csr_idx_base);
    hipsparseSetMatIndexBase(bsr_descr, bsr_idx_base);

    // Correct M and N to ensure that they are divisible by the block dimension
    m = mb * block_dim;
    n = nb * block_dim;

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0 || block_dim <= 0)
    {
#ifdef __HIP_PLATFORM_NVCC__
        // Do not test args in cusparse
        return HIPSPARSE_STATUS_SUCCESS;
#endif
        auto bsr_row_ptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto bsr_col_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto bsr_val_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto csr_row_ptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto csr_col_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto csr_val_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* bsr_row_ptr = (int*)bsr_row_ptr_managed.get();
        int* bsr_col_ind = (int*)bsr_col_ind_managed.get();
        T*   bsr_val     = (T*)bsr_val_managed.get();
        int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
        int* csr_col_ind = (int*)csr_col_ind_managed.get();
        T*   csr_val     = (T*)csr_val_managed.get();

        if(!csr_row_ptr || !csr_col_ind || !csr_val || !csr_row_ptr || !csr_col_ind || !csr_val)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!bsr_row_ptr || !bsr_col_ind || !bsr_val || "
                                            "!csr_row_ptr || !csr_col_ind || !csr_val");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        status = hipsparseXbsr2csr(handle,
                                   dir,
                                   m,
                                   n,
                                   bsr_descr,
                                   bsr_val,
                                   bsr_row_ptr,
                                   bsr_col_ind,
                                   block_dim,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);

        if(m < 0 || n < 0 || nnz < 0 || block_dim < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || nnz < 0 || block_dim < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0 && block_dim >= 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Host structures
    std::vector<int> hbsr_row_ptr;
    std::vector<int> hbsr_col_ind;
    std::vector<T>   hbsr_val;

    int nnzb;
    // // Sample initial COO matrix on CPU
    // srand(12345ULL);
    // if(binfile != "")
    // {
    //     if(read_bin_matrix(
    //            binfile.c_str(), m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base)
    //        != 0)
    //     {
    //         fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
    //         return HIPSPARSE_STATUS_INTERNAL_ERROR;
    //     }
    // }
    // else if(argus.laplacian)
    // {
    //     m = n = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base);
    //     nnz   = hcsr_row_ptr[m];
    // }
    // else
    // {
    //     std::vector<int> hcoo_row_ind;

    //     if(filename != "")
    //     {
    //         if(read_mtx_matrix(
    //                filename.c_str(), m, n, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, idx_base)
    //            != 0)
    //         {
    //             fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
    //             return HIPSPARSE_STATUS_INTERNAL_ERROR;
    //         }
    //     }
    //     else
    //     {
    //         gen_matrix_coo(m, n, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, idx_base);
    //     }

    //     // Convert COO to CSR
    //     hcsr_row_ptr.resize(m + 1, 0);
    //     for(int i = 0; i < nnz; ++i)
    //     {
    //         ++hcsr_row_ptr[hcoo_row_ind[i] + 1 - idx_base];
    //     }

    //     hcsr_row_ptr[0] = idx_base;
    //     for(int i = 0; i < m; ++i)
    //     {
    //         hcsr_row_ptr[i + 1] += hcsr_row_ptr[i];
    //     }
    // }

    // Mb and Nb can be modified by rocsparse_init_bsr_matrix
    m = mb * block_dim;
    n = nb * block_dim;

    // Allocate memory on the device
    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};
    auto dbsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnzb), device_free};
    auto dbsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnzb * block_dim * block_dim), device_free};
    auto dcsr_row_ptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();
    int* dbsr_col_ind = (int*)dbsr_col_ind_managed.get();
    T*   dbsr_val     = (T*)dbsr_val_managed.get();
    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val     = (T*)dcsr_val_managed.get();

    if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || "
                                        "!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_row_ptr, hbsr_row_ptr.data(), sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind.data(), sizeof(int) * nnzb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_val, hbsr_val.data(), sizeof(T) * nnzb * block_dim * block_dim, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseXbsr2csr(handle,
                                                dir,
                                                m,
                                                n,
                                                bsr_descr,
                                                dbsr_val,
                                                dbsr_row_ptr,
                                                dbsr_col_ind,
                                                block_dim,
                                                csr_descr,
                                                dcsr_val,
                                                dcsr_row_ptr,
                                                dcsr_col_ind));

        // Copy output from device to host
        std::vector<int> hcsr_row_ptr(m + 1);
        std::vector<int> hcsr_col_ind(nnzb * block_dim * block_dim);
        std::vector<T>   hcsr_val(nnzb * block_dim * block_dim);

        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_row_ptr.data(), dcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind.data(), dcsr_col_ind, sizeof(int) * nnzb * block_dim * block_dim, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val.data(), dcsr_val, sizeof(T) * nnzb * block_dim * block_dim, hipMemcpyDeviceToHost));

        // Host bsr2csr conversion
        std::vector<int> hcsr_row_ptr_gold(m + 1);
        std::vector<int> hcsr_col_ind_gold(nnzb * block_dim * block_dim, 0);
        std::vector<T>   hcsr_val_gold(nnzb * block_dim * block_dim);

        // Host bsr2csr goes here

        // Unit check
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_gold.data(), hcsr_row_ptr.data());
        unit_check_general(1, nnzb * block_dim * block_dim, 1, hcsr_col_ind_gold.data(), hcsr_col_ind.data());
        unit_check_general(1, nnzb * block_dim * block_dim, 1, hcsr_val_gold.data(), hcsr_val.data());
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_BSR2CSR_HPP
