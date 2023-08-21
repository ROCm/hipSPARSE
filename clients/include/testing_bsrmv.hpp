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
#ifndef TESTING_BSRMV_HPP
#define TESTING_BSRMV_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <cmath>
#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_bsrmv_bad_arg(void)
{
#if(!defined(CUDART_VERSION))

    int                  safe_size = 100;
    int                  safe_dim  = 2;
    T                    alpha     = 0.6;
    T                    beta      = 0.2;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseDirection_t dirA      = HIPSPARSE_DIRECTION_COLUMN;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dy_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* dptr = (int*)dptr_managed.get();
    int* dcol = (int*)dcol_managed.get();
    T*   dval = (T*)dval_managed.get();
    T*   dx   = (T*)dx_managed.get();
    T*   dy   = (T*)dy_managed.get();

    if(!dval || !dptr || !dcol || !dx || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test hipsparseXbsrmv
    verify_hipsparse_status_invalid_handle(hipsparseXbsrmv(nullptr,
                                                           dirA,
                                                           transA,
                                                           safe_size,
                                                           safe_size,
                                                           safe_size,
                                                           &alpha,
                                                           descr,
                                                           dval,
                                                           dptr,
                                                           dcol,
                                                           safe_dim,
                                                           dx,
                                                           &beta,
                                                           dy));
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
                                                            safe_size,
                                                            (T*)nullptr,
                                                            descr,
                                                            dval,
                                                            dptr,
                                                            dcol,
                                                            safe_dim,
                                                            dx,
                                                            &beta,
                                                            dy),
                                            "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            nullptr,
                                                            dval,
                                                            dptr,
                                                            dcol,
                                                            safe_dim,
                                                            dx,
                                                            &beta,
                                                            dy),
                                            "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            descr,
                                                            (T*)nullptr,
                                                            dptr,
                                                            dcol,
                                                            safe_dim,
                                                            dx,
                                                            &beta,
                                                            dy),
                                            "Error: bsr_val is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            descr,
                                                            dval,
                                                            nullptr,
                                                            dcol,
                                                            safe_dim,
                                                            dx,
                                                            &beta,
                                                            dy),
                                            "Error: bsr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            descr,
                                                            dval,
                                                            dptr,
                                                            nullptr,
                                                            safe_dim,
                                                            dx,
                                                            &beta,
                                                            dy),
                                            "Error: bsr_col_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            descr,
                                                            dval,
                                                            dptr,
                                                            dcol,
                                                            safe_dim,
                                                            (T*)nullptr,
                                                            &beta,
                                                            dy),
                                            "Error: xy is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            descr,
                                                            dval,
                                                            dptr,
                                                            dcol,
                                                            safe_dim,
                                                            dx,
                                                            (T*)nullptr,
                                                            dy),
                                            "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            descr,
                                                            dval,
                                                            dptr,
                                                            dcol,
                                                            safe_dim,
                                                            dx,
                                                            &beta,
                                                            (T*)nullptr),
                                            "Error: y is nullptr");
    verify_hipsparse_status_invalid_size(hipsparseXbsrmv(handle,
                                                         dirA,
                                                         transA,
                                                         -1,
                                                         safe_size,
                                                         safe_size,
                                                         &alpha,
                                                         descr,
                                                         dval,
                                                         dptr,
                                                         dcol,
                                                         safe_dim,
                                                         dx,
                                                         &beta,
                                                         dy),
                                         "Error: mb is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXbsrmv(handle,
                                                         dirA,
                                                         transA,
                                                         safe_size,
                                                         -1,
                                                         safe_size,
                                                         &alpha,
                                                         descr,
                                                         dval,
                                                         dptr,
                                                         dcol,
                                                         safe_dim,
                                                         dx,
                                                         &beta,
                                                         dy),
                                         "Error: nb is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXbsrmv(handle,
                                                         dirA,
                                                         transA,
                                                         safe_size,
                                                         safe_size,
                                                         -1,
                                                         &alpha,
                                                         descr,
                                                         dval,
                                                         dptr,
                                                         dcol,
                                                         safe_dim,
                                                         dx,
                                                         &beta,
                                                         dy),
                                         "Error: nnzb is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXbsrmv(handle,
                                                         dirA,
                                                         transA,
                                                         safe_size,
                                                         safe_size,
                                                         safe_size,
                                                         &alpha,
                                                         descr,
                                                         dval,
                                                         dptr,
                                                         dcol,
                                                         -1,
                                                         dx,
                                                         &beta,
                                                         dy),
                                         "Error: block_dim is invalid");
#endif
}

template <typename T>
hipsparseStatus_t testing_bsrmv(Arguments argus)
{
    int                  safe_size = 100;
    int                  m         = argus.M;
    int                  n         = argus.N;
    int                  block_dim = argus.block_dim;
    T                    h_alpha   = make_DataType<T>(argus.alpha);
    T                    h_beta    = make_DataType<T>(argus.beta);
    hipsparseOperation_t transA    = argus.transA;
    hipsparseIndexBase_t idx_base  = argus.idx_base;
    hipsparseDirection_t dir       = argus.dirA;
    std::string          binfile   = "";
    std::string          filename  = "";

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

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr(new descr_struct);
    hipsparseMatDescr_t           descr = test_descr->descr;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    int mb = (m + block_dim - 1) / block_dim;
    int nb = (n + block_dim - 1) / block_dim;

    if(block_dim == 1)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        // cusparse only accepts block_dim > 1
        return HIPSPARSE_STATUS_SUCCESS;
#endif
    }

    // Argument sanity check before allocating invalid memory
    if(mb == 0 || nb == 0)
    {
        auto dptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dcol_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dy_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* dptr = (int*)dptr_managed.get();
        int* dcol = (int*)dcol_managed.get();
        T*   dval = (T*)dval_managed.get();
        T*   dx   = (T*)dx_managed.get();
        T*   dy   = (T*)dy_managed.get();

        if(!dval || !dptr || !dcol || !dx || !dy)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!dptr || !dcol || !dval || !dx || !dy");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        hipsparseStatus_t status = hipsparseXbsrmv(handle,
                                 dir,
                                 transA,
                                 mb,
                                 nb,
                                 safe_size,
                                 &h_alpha,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol,
                                 block_dim,
                                 dx,
                                 &h_beta,
                                 dy);

        verify_hipsparse_status_success(status, "mb >= 0 && nb >= 0");

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;
    int              nnz;

    // Initial Data on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(
               binfile.c_str(), m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }
    else if(argus.laplacian)
    {
        m = n = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base);
        nnz   = hcsr_row_ptr[m];
    }
    else
    {
        std::vector<int> coo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(
                   filename.c_str(), m, n, nnz, coo_row_ind, hcsr_col_ind, hcsr_val, idx_base)
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
            gen_matrix_coo(m, n, nnz, coo_row_ind, hcsr_col_ind, hcsr_val, idx_base);
        }

        // Convert COO to CSR
        hcsr_row_ptr.resize(m + 1, 0);
        for(int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptr[coo_row_ind[i] + 1 - idx_base];
        }

        hcsr_row_ptr[0] = idx_base;
        for(int i = 0; i < m; ++i)
        {
            hcsr_row_ptr[i + 1] += hcsr_row_ptr[i];
        }
    }

    mb = (m + block_dim - 1) / block_dim;
    nb = (n + block_dim - 1) / block_dim;

    std::vector<T> hx(nb * block_dim);
    std::vector<T> hy_1(mb * block_dim);
    std::vector<T> hy_2(mb * block_dim);
    std::vector<T> hy_gold(mb * block_dim);

    hipsparseInit<T>(hx, 1, nb * block_dim);
    hipsparseInit<T>(hy_1, 1, mb * block_dim);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};
    auto dx_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nb * block_dim), device_free};
    auto dy_1_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * mb * block_dim), device_free};
    auto dy_2_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * mb * block_dim), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val     = (T*)dcsr_val_managed.get();
    int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();
    T*   dx           = (T*)dx_managed.get();
    T*   dy_1         = (T*)dy_1_managed.get();
    T*   dy_2         = (T*)dy_2_managed.get();
    T*   d_alpha      = (T*)d_alpha_managed.get();
    T*   d_beta       = (T*)d_beta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Convert to BSR
    int nnzb;
    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsrNnz(handle,
                                               dir,
                                               m,
                                               n,
                                               descr,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               block_dim,
                                               descr,
                                               dbsr_row_ptr,
                                               &nnzb));

    auto dbsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnzb), device_free};
    auto dbsr_val_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(T) * nnzb * block_dim * block_dim), device_free};

    int* dbsr_col_ind = (int*)dbsr_col_ind_managed.get();
    T*   dbsr_val     = (T*)dbsr_val_managed.get();

    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsr(handle,
                                            dir,
                                            m,
                                            n,
                                            descr,
                                            dcsr_val,
                                            dcsr_row_ptr,
                                            dcsr_col_ind,
                                            block_dim,
                                            descr,
                                            dbsr_val,
                                            dbsr_row_ptr,
                                            dbsr_col_ind));

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * m, hipMemcpyHostToDevice));

        // ROCSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXbsrmv(handle,
                                              dir,
                                              transA,
                                              mb,
                                              nb,
                                              nnzb,
                                              &h_alpha,
                                              descr,
                                              dbsr_val,
                                              dbsr_row_ptr,
                                              dbsr_col_ind,
                                              block_dim,
                                              dx,
                                              &h_beta,
                                              dy_1));

        // ROCSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXbsrmv(handle,
                                              dir,
                                              transA,
                                              mb,
                                              nb,
                                              nnzb,
                                              d_alpha,
                                              descr,
                                              dbsr_val,
                                              dbsr_row_ptr,
                                              dbsr_col_ind,
                                              block_dim,
                                              dx,
                                              d_beta,
                                              dy_2));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * m, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * m, hipMemcpyDeviceToHost));

        // Host bsrmv
        std::vector<int> hbsr_row_ptr(mb + 1);
        std::vector<int> hbsr_col_ind(nnzb);
        std::vector<T>   hbsr_val(nnzb * block_dim * block_dim);

        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_row_ptr.data(), dbsr_row_ptr, sizeof(int) * (mb + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_col_ind.data(), dbsr_col_ind, sizeof(int) * nnzb, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val.data(),
                                  dbsr_val,
                                  sizeof(T) * nnzb * block_dim * block_dim,
                                  hipMemcpyDeviceToHost));

        host_bsrmv(dir,
                   transA,
                   mb,
                   nb,
                   nnzb,
                   h_alpha,
                   hbsr_row_ptr.data(),
                   hbsr_col_ind.data(),
                   hbsr_val.data(),
                   block_dim,
                   hx.data(),
                   h_beta,
                   hy_gold.data(),
                   idx_base);

        unit_check_near(1, m, 1, hy_gold.data(), hy_1.data());
        unit_check_near(1, m, 1, hy_gold.data(), hy_2.data());
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_BSRMV_HPP
