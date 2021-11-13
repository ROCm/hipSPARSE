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
#ifndef TESTING_HYB2CSR_HPP
#define TESTING_HYB2CSR_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse/hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

struct test_hyb
{
    int                     m;
    int                     n;
    hipsparseHybPartition_t partition;
    int                     ell_nnz;
    int                     ell_width;
    int*                    ell_col_ind;
    void*                   ell_val;
    int                     coo_nnz;
    int*                    coo_row_ind;
    int*                    coo_col_ind;
    void*                   coo_val;
};

template <typename T>
void testing_hyb2csr_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVCC__
    // do not test for bad args
    return;
#endif
    int               safe_size = 100;
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<hyb_struct> unique_ptr_hyb(new hyb_struct);
    hipsparseHybMat_t           hyb = unique_ptr_hyb->hyb;

    test_hyb* dhyb = (test_hyb*)hyb;

    dhyb->m       = safe_size;
    dhyb->n       = safe_size;
    dhyb->ell_nnz = safe_size;
    dhyb->coo_nnz = safe_size;

    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* csr_col_ind = (int*)csr_col_ind_managed.get();
    T*   csr_val     = (T*)csr_val_managed.get();

    if(!csr_row_ptr || !csr_col_ind || !csr_val)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing hipsparseXhyb2csr()

    // Testing for (csr_row_ptr == nullptr)
    {
        int* csr_row_ptr_null = nullptr;

        status = hipsparseXhyb2csr(handle, descr, hyb, csr_val, csr_row_ptr_null, csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    // Testing for (csr_col_ind == nullptr)
    {
        int* csr_col_ind_null = nullptr;

        status = hipsparseXhyb2csr(handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }

    // Testing for (csr_val == nullptr)
    {
        T* csr_val_null = nullptr;

        status = hipsparseXhyb2csr(handle, descr, hyb, csr_val_null, csr_row_ptr, csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");
    }

    // Testing for (descr == nullptr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXhyb2csr(handle, descr_null, hyb, csr_val, csr_row_ptr, csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");
    }

    // Testing for (hyb == nullptr)
    {
        hipsparseHybMat_t* hyb_null = nullptr;

        status = hipsparseXhyb2csr(handle, descr, hyb_null, csr_val, csr_row_ptr, csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXhyb2csr(handle_null, descr, hyb, csr_val, csr_row_ptr, csr_col_ind);
        verify_hipsparse_status_invalid_handle(status);
    }
}

template <typename T>
hipsparseStatus_t testing_hyb2csr(Arguments argus)
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

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<hyb_struct> unique_ptr_hyb(new hyb_struct);
    hipsparseHybMat_t           hyb = unique_ptr_hyb->hyb;

    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0)
    {
#ifdef __HIP_PLATFORM_NVCC__
        // Do not test args in cusparse
        return HIPSPARSE_STATUS_SUCCESS;
#endif
        test_hyb* dhyb = (test_hyb*)hyb;

        dhyb->m       = m;
        dhyb->n       = n;
        dhyb->ell_nnz = safe_size;
        dhyb->coo_nnz = safe_size;

        auto csr_row_ptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto csr_col_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto csr_val_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
        int* csr_col_ind = (int*)csr_col_ind_managed.get();
        T*   csr_val     = (T*)csr_val_managed.get();

        if(!csr_row_ptr || !csr_col_ind || !csr_val)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!csr_row_ptr || !csr_col_ind || !csr_val");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        status = hipsparseXhyb2csr(handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind);

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

    // Host structures
    std::vector<int> hcsr_row_ptr_gold;
    std::vector<int> hcsr_col_ind_gold;
    std::vector<T>   hcsr_val_gold;

    // Sample initial COO matrix on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(binfile.c_str(),
                           m,
                           n,
                           nnz,
                           hcsr_row_ptr_gold,
                           hcsr_col_ind_gold,
                           hcsr_val_gold,
                           idx_base)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }
    else if(argus.laplacian)
    {
        m = n = gen_2d_laplacian(
            argus.laplacian, hcsr_row_ptr_gold, hcsr_col_ind_gold, hcsr_val_gold, idx_base);
        nnz = hcsr_row_ptr_gold[m];
    }
    else
    {
        std::vector<int> hcoo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(filename.c_str(),
                               m,
                               n,
                               nnz,
                               hcoo_row_ind,
                               hcsr_col_ind_gold,
                               hcsr_val_gold,
                               idx_base)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            gen_matrix_coo(m, n, nnz, hcoo_row_ind, hcsr_col_ind_gold, hcsr_val_gold, idx_base);
        }

        // Convert COO to CSR
        hcsr_row_ptr_gold.resize(m + 1, 0);
        for(int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptr_gold[hcoo_row_ind[i] + 1 - idx_base];
        }

        hcsr_row_ptr_gold[0] = idx_base;
        for(int i = 0; i < m; ++i)
        {
            hcsr_row_ptr_gold[i + 1] += hcsr_row_ptr_gold[i];
        }
    }

    // Allocate memory on the device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val     = (T*)dcsr_val_managed.get();

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr_gold.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind_gold.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val, hcsr_val_gold.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Convert CSR to HYB
    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2hyb(handle,
                                            m,
                                            n,
                                            descr,
                                            dcsr_val,
                                            dcsr_row_ptr,
                                            dcsr_col_ind,
                                            hyb,
                                            0,
                                            HIPSPARSE_HYB_PARTITION_AUTO));

    // Set all CSR arrays to zero
    CHECK_HIP_ERROR(hipMemset(dcsr_row_ptr, 0, sizeof(int) * (m + 1)));
    CHECK_HIP_ERROR(hipMemset(dcsr_col_ind, 0, sizeof(int) * nnz));
    CHECK_HIP_ERROR(hipMemset(dcsr_val, 0, sizeof(T) * nnz));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(
            hipsparseXhyb2csr(handle, descr, hyb, dcsr_val, dcsr_row_ptr, dcsr_col_ind));

        // Copy output from device to host
        std::vector<int> hcsr_row_ptr(m + 1);
        std::vector<int> hcsr_col_ind(nnz);
        std::vector<T>   hcsr_val(nnz);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr.data(), dcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_col_ind.data(), dcsr_col_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val.data(), dcsr_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_gold.data(), hcsr_row_ptr.data());
        unit_check_general(1, nnz, 1, hcsr_col_ind_gold.data(), hcsr_col_ind.data());
        unit_check_general(1, nnz, 1, hcsr_val_gold.data(), hcsr_val.data());
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_HYB2CSR_HPP
