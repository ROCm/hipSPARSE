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
#ifndef TESTING_CSRSM2_HPP
#define TESTING_CSRSM2_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <cmath>
#include <hipsparse.h>
#include <limits>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_csrsm2_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                    m         = 100;
    int                    nrhs      = 100;
    int                    nnz       = 100;
    int                    safe_size = 100;
    T                      h_alpha   = 0.6;
    hipsparseOperation_t   transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t   transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseSolvePolicy_t policy    = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
    hipsparseStatus_t      status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<csrsm2_struct> unique_ptr_csrsm2_info(new csrsm2_struct);
    csrsm2Info_t                   info = unique_ptr_csrsm2_info->info;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*  dptr    = (int*)dptr_managed.get();
    int*  dcol    = (int*)dcol_managed.get();
    T*    dval    = (T*)dval_managed.get();
    T*    dB      = (T*)dB_managed.get();
    void* dbuffer = (void*)dbuffer_managed.get();

    if(!dval || !dptr || !dcol || !dB || !dbuffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing hipsparseXcsrsm2_bufferSizeExt
    size_t size;

    // testing for(nullptr == alpha)
    {
        T* dalpha_null = nullptr;

        status = hipsparseXcsrsm2_bufferSizeExt(handle,
                                                0,
                                                transA,
                                                transB,
                                                m,
                                                nrhs,
                                                nnz,
                                                dalpha_null,
                                                descr,
                                                dval,
                                                dptr,
                                                dcol,
                                                dB,
                                                safe_size,
                                                info,
                                                policy,
                                                &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dalpha is nullptr");
    }
    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXcsrsm2_bufferSizeExt(handle,
                                                0,
                                                transA,
                                                transB,
                                                m,
                                                nrhs,
                                                nnz,
                                                &h_alpha,
                                                descr,
                                                dval,
                                                dptr_null,
                                                dcol,
                                                dB,
                                                safe_size,
                                                info,
                                                policy,
                                                &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXcsrsm2_bufferSizeExt(handle,
                                                0,
                                                transA,
                                                transB,
                                                m,
                                                nrhs,
                                                nnz,
                                                &h_alpha,
                                                descr,
                                                dval,
                                                dptr,
                                                dcol_null,
                                                dB,
                                                safe_size,
                                                info,
                                                policy,
                                                &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXcsrsm2_bufferSizeExt(handle,
                                                0,
                                                transA,
                                                transB,
                                                m,
                                                nrhs,
                                                nnz,
                                                &h_alpha,
                                                descr,
                                                dval_null,
                                                dptr,
                                                dcol,
                                                dB,
                                                safe_size,
                                                info,
                                                policy,
                                                &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == buffer_size)
    {
        size_t* size_null = nullptr;

        status = hipsparseXcsrsm2_bufferSizeExt(handle,
                                                0,
                                                transA,
                                                transB,
                                                m,
                                                nrhs,
                                                nnz,
                                                &h_alpha,
                                                descr,
                                                dval,
                                                dptr,
                                                dcol,
                                                dB,
                                                safe_size,
                                                info,
                                                policy,
                                                size_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: size is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXcsrsm2_bufferSizeExt(handle,
                                                0,
                                                transA,
                                                transB,
                                                m,
                                                nrhs,
                                                nnz,
                                                &h_alpha,
                                                descr_null,
                                                dval,
                                                dptr,
                                                dcol,
                                                dB,
                                                safe_size,
                                                info,
                                                policy,
                                                &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        csrsm2Info_t info_null = nullptr;

        status = hipsparseXcsrsm2_bufferSizeExt(handle,
                                                0,
                                                transA,
                                                transB,
                                                m,
                                                nrhs,
                                                nnz,
                                                &h_alpha,
                                                descr,
                                                dval,
                                                dptr,
                                                dcol,
                                                dB,
                                                safe_size,
                                                info_null,
                                                policy,
                                                &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == B)
    {
        T* B_null = nullptr;

        status = hipsparseXcsrsm2_bufferSizeExt(handle,
                                                0,
                                                transA,
                                                transB,
                                                m,
                                                nrhs,
                                                nnz,
                                                &h_alpha,
                                                descr,
                                                dval,
                                                dptr,
                                                dcol,
                                                B_null,
                                                safe_size,
                                                info,
                                                policy,
                                                &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dB is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrsm2_bufferSizeExt(handle_null,
                                                0,
                                                transA,
                                                transB,
                                                m,
                                                nrhs,
                                                nnz,
                                                &h_alpha,
                                                descr,
                                                dval,
                                                dptr,
                                                dcol,
                                                dB,
                                                safe_size,
                                                info,
                                                policy,
                                                &size);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing hipsparseXcsrsm2_analysis

    // testing for(nullptr == dalpha)
    {
        T* dalpha_null = nullptr;

        status = hipsparseXcsrsm2_analysis(handle,
                                           0,
                                           transA,
                                           transB,
                                           m,
                                           nrhs,
                                           nnz,
                                           dalpha_null,
                                           descr,
                                           dval,
                                           dptr,
                                           dcol,
                                           dB,
                                           safe_size,
                                           info,
                                           policy,
                                           dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dalpha is nullptr");
    }
    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXcsrsm2_analysis(handle,
                                           0,
                                           transA,
                                           transB,
                                           m,
                                           nrhs,
                                           nnz,
                                           &h_alpha,
                                           descr,
                                           dval,
                                           dptr_null,
                                           dcol,
                                           dB,
                                           safe_size,
                                           info,
                                           policy,
                                           dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXcsrsm2_analysis(handle,
                                           0,
                                           transA,
                                           transB,
                                           m,
                                           nrhs,
                                           nnz,
                                           &h_alpha,
                                           descr,
                                           dval,
                                           dptr,
                                           dcol_null,
                                           dB,
                                           safe_size,
                                           info,
                                           policy,
                                           dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXcsrsm2_analysis(handle,
                                           0,
                                           transA,
                                           transB,
                                           m,
                                           nrhs,
                                           nnz,
                                           &h_alpha,
                                           descr,
                                           dval_null,
                                           dptr,
                                           dcol,
                                           dB,
                                           safe_size,
                                           info,
                                           policy,
                                           dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        void* dbuffer_null = nullptr;

        status = hipsparseXcsrsm2_analysis(handle,
                                           0,
                                           transA,
                                           transB,
                                           m,
                                           nrhs,
                                           nnz,
                                           &h_alpha,
                                           descr,
                                           dval,
                                           dptr,
                                           dcol,
                                           dB,
                                           safe_size,
                                           info,
                                           policy,
                                           dbuffer_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXcsrsm2_analysis(handle,
                                           0,
                                           transA,
                                           transB,
                                           m,
                                           nrhs,
                                           nnz,
                                           &h_alpha,
                                           descr_null,
                                           dval,
                                           dptr,
                                           dcol,
                                           dB,
                                           safe_size,
                                           info,
                                           policy,
                                           dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == dB)
    {
        T* dB_null = nullptr;

        status = hipsparseXcsrsm2_analysis(handle,
                                           0,
                                           transA,
                                           transB,
                                           m,
                                           nrhs,
                                           nnz,
                                           &h_alpha,
                                           descr,
                                           dval,
                                           dptr,
                                           dcol,
                                           dB_null,
                                           safe_size,
                                           info,
                                           policy,
                                           dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dB is nullptr");
    }
    // testing for(nullptr == info)
    {
        csrsm2Info_t info_null = nullptr;

        status = hipsparseXcsrsm2_analysis(handle,
                                           0,
                                           transA,
                                           transB,
                                           m,
                                           nrhs,
                                           nnz,
                                           &h_alpha,
                                           descr,
                                           dval,
                                           dptr,
                                           dcol,
                                           dB,
                                           safe_size,
                                           info_null,
                                           policy,
                                           dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrsm2_analysis(handle_null,
                                           0,
                                           transA,
                                           transB,
                                           m,
                                           nrhs,
                                           nnz,
                                           &h_alpha,
                                           descr,
                                           dval,
                                           dptr,
                                           dcol,
                                           dB,
                                           safe_size,
                                           info,
                                           policy,
                                           dbuffer);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing rocsparse_csrsm2

    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXcsrsm2_solve(handle,
                                        0,
                                        transA,
                                        transB,
                                        m,
                                        nrhs,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval,
                                        dptr_null,
                                        dcol,
                                        dB,
                                        safe_size,
                                        info,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXcsrsm2_solve(handle,
                                        0,
                                        transA,
                                        transB,
                                        m,
                                        nrhs,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval,
                                        dptr,
                                        dcol_null,
                                        dB,
                                        safe_size,
                                        info,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXcsrsm2_solve(handle,
                                        0,
                                        transA,
                                        transB,
                                        m,
                                        nrhs,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval_null,
                                        dptr,
                                        dcol,
                                        dB,
                                        safe_size,
                                        info,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == dB)
    {
        T* dB_null = nullptr;

        status = hipsparseXcsrsm2_solve(handle,
                                        0,
                                        transA,
                                        transB,
                                        m,
                                        nrhs,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval,
                                        dptr,
                                        dcol,
                                        dB_null,
                                        safe_size,
                                        info,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dB is nullptr");
    }
    // testing for(nullptr == d_alpha)
    {
        T* d_alpha_null = nullptr;

        status = hipsparseXcsrsm2_solve(handle,
                                        0,
                                        transA,
                                        transB,
                                        m,
                                        nrhs,
                                        nnz,
                                        d_alpha_null,
                                        descr,
                                        dval,
                                        dptr,
                                        dcol,
                                        dB,
                                        safe_size,
                                        info,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: alpha is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        void* dbuffer_null = nullptr;

        status = hipsparseXcsrsm2_solve(handle,
                                        0,
                                        transA,
                                        transB,
                                        m,
                                        nrhs,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval,
                                        dptr,
                                        dcol,
                                        dB,
                                        safe_size,
                                        info,
                                        policy,
                                        dbuffer_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXcsrsm2_solve(handle,
                                        0,
                                        transA,
                                        transB,
                                        m,
                                        nrhs,
                                        nnz,
                                        &h_alpha,
                                        descr_null,
                                        dval,
                                        dptr,
                                        dcol,
                                        dB,
                                        safe_size,
                                        info,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        csrsm2Info_t info_null = nullptr;

        status = hipsparseXcsrsm2_solve(handle,
                                        0,
                                        transA,
                                        transB,
                                        m,
                                        nrhs,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval,
                                        dptr,
                                        dcol,
                                        dB,
                                        safe_size,
                                        info_null,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrsm2_solve(handle_null,
                                        0,
                                        transA,
                                        transB,
                                        m,
                                        nrhs,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval,
                                        dptr,
                                        dcol,
                                        dB,
                                        safe_size,
                                        info,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing hipsparseXcsrsm2_zeroPivot
    int position;

    // testing for(nullptr == position)
    {
        int* position_null = nullptr;

        status = hipsparseXcsrsm2_zeroPivot(handle, info, position_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: position is nullptr");
    }
    // testing for(nullptr == info)
    {
        csrsm2Info_t info_null = nullptr;

        status = hipsparseXcsrsm2_zeroPivot(handle, info_null, &position);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrsm2_zeroPivot(handle_null, info, &position);
        verify_hipsparse_status_invalid_handle(status);
    }
#endif
}


template <typename T>
hipsparseStatus_t testing_csrsm2(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    int                    m        = argus.M;
    int                    nrhs     = argus.N;
    hipsparseIndexBase_t   idx_base = argus.idx_base;
    hipsparseOperation_t   transA   = argus.transA;
    hipsparseOperation_t   transB   = argus.transB;
    hipsparseDiagType_t    diag     = argus.diag_type;
    hipsparseFillMode_t    uplo     = argus.fill_mode;
    hipsparseSolvePolicy_t policy   = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
    T                      h_alpha  = make_DataType<T>(argus.alpha);
    std::string            filename = argus.filename;

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr(new descr_struct);
    hipsparseMatDescr_t           descr = test_descr->descr;

    std::unique_ptr<csrsm2_struct> unique_ptr_csrsm2_info(new csrsm2_struct);
    csrsm2Info_t                   info = unique_ptr_csrsm2_info->info;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    // Set matrix diag type
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatDiagType(descr, diag));

    // Set matrix fill mode
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatFillMode(descr, uplo));

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_csr_matrix(filename, m, m, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    int ldb = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : nrhs;

    std::vector<T> hB_1(m * nrhs);
    std::vector<T> hB_2(m * nrhs);
    std::vector<T> hB_gold(m * nrhs);

    int h_analysis_pivot_gold;
    int h_analysis_pivot_1;
    int h_analysis_pivot_2;
    int h_solve_pivot_gold;
    int h_solve_pivot_1;
    int h_solve_pivot_2;

    hipsparseInit<T>(hB_1, 1, m * nrhs);
    hB_2    = hB_1;
    hB_gold = hB_1;

    // Allocate memory on device
    auto dptr_managed    = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcol_managed    = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dval_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dB_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * nrhs), device_free};
    auto dB_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * nrhs), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_analysis_pivot_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
    auto d_solve_pivot_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};

    int* dptr               = (int*)dptr_managed.get();
    int* dcol               = (int*)dcol_managed.get();
    T*   dval               = (T*)dval_managed.get();
    T*   dB_1               = (T*)dB_1_managed.get();
    T*   dB_2               = (T*)dB_2_managed.get();
    T*   d_alpha            = (T*)d_alpha_managed.get();
    int* d_analysis_pivot_2 = (int*)d_analysis_pivot_2_managed.get();
    int* d_solve_pivot_2    = (int*)d_solve_pivot_2_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB_1, hB_1.data(), sizeof(T) * m * nrhs, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // Obtain csrsm2 buffer size
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrsm2_bufferSizeExt(handle,
                                                         0,
                                                         transA,
                                                         transB,
                                                         m,
                                                         nrhs,
                                                         nnz,
                                                         &h_alpha,
                                                         descr,
                                                         dval,
                                                         dptr,
                                                         dcol,
                                                         dB_1,
                                                         ldb,
                                                         info,
                                                         policy,
                                                         &bufferSize));

    // Allocate buffer on the device
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * bufferSize), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    if(argus.unit_check)
    {
        hipsparseStatus_t status_analysis_1;
        hipsparseStatus_t status_analysis_2;
        hipsparseStatus_t status_solve_1;
        hipsparseStatus_t status_solve_2;

        CHECK_HIP_ERROR(hipMemcpy(dB_2, hB_2.data(), sizeof(T) * m * nrhs, hipMemcpyHostToDevice));

        // csrsm2 analysis - host mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrsm2_analysis(handle,
                                                        0,
                                                        transA,
                                                        transB,
                                                        m,
                                                        nrhs,
                                                        nnz,
                                                        &h_alpha,
                                                        descr,
                                                        dval,
                                                        dptr,
                                                        dcol,
                                                        dB_1,
                                                        ldb,
                                                        info,
                                                        policy,
                                                        dbuffer));

        // Get pivot
        status_analysis_1 = hipsparseXcsrsm2_zeroPivot(handle, info, &h_analysis_pivot_1);
        if(h_analysis_pivot_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(status_analysis_1,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        }

        // csrsm2 analysis - device mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrsm2_analysis(handle,
                                                        0,
                                                        transA,
                                                        transB,
                                                        m,
                                                        nrhs,
                                                        nnz,
                                                        d_alpha,
                                                        descr,
                                                        dval,
                                                        dptr,
                                                        dcol,
                                                        dB_2,
                                                        ldb,
                                                        info,
                                                        policy,
                                                        dbuffer));

        // Get pivot
        status_analysis_2 = hipsparseXcsrsm2_zeroPivot(handle, info, d_analysis_pivot_2);
        if(h_analysis_pivot_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(status_analysis_2,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        }

        // csrsm2 solve - host mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrsm2_solve(handle,
                                                     0,
                                                     transA,
                                                     transB,
                                                     m,
                                                     nrhs,
                                                     nnz,
                                                     &h_alpha,
                                                     descr,
                                                     dval,
                                                     dptr,
                                                     dcol,
                                                     dB_1,
                                                     ldb,
                                                     info,
                                                     policy,
                                                     dbuffer));

        // Get pivot
        status_solve_1 = hipsparseXcsrsm2_zeroPivot(handle, info, &h_solve_pivot_1);
        if(h_solve_pivot_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(status_solve_1,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        }

        // csrsm2 solve - device mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrsm2_solve(handle,
                                                     0,
                                                     transA,
                                                     transB,
                                                     m,
                                                     nrhs,
                                                     nnz,
                                                     d_alpha,
                                                     descr,
                                                     dval,
                                                     dptr,
                                                     dcol,
                                                     dB_2,
                                                     ldb,
                                                     info,
                                                     policy,
                                                     dbuffer));

        // Get pivot
        status_solve_2 = hipsparseXcsrsm2_zeroPivot(handle, info, d_solve_pivot_2);
        if(h_solve_pivot_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(status_solve_2,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        }

        // Copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hB_1.data(), dB_1, sizeof(T) * m * nrhs, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hB_2.data(), dB_2, sizeof(T) * m * nrhs, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(&h_analysis_pivot_2, d_analysis_pivot_2, sizeof(int), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(&h_solve_pivot_2, d_solve_pivot_2, sizeof(int), hipMemcpyDeviceToHost));

        // Host csrsm2
        host_csrsm(m,
                   nrhs,
                   nnz,
                   transA,
                   transB,
                   h_alpha,
                   hcsr_row_ptr,
                   hcsr_col_ind,
                   hcsr_val,
                   hB_gold,
                   ldb,
                   diag,
                   uplo,
                   idx_base,
                   &h_analysis_pivot_gold,
                   &h_solve_pivot_gold);

        // Check pivots
        unit_check_general(1, 1, 1, &h_analysis_pivot_gold, &h_analysis_pivot_1);
        unit_check_general(1, 1, 1, &h_analysis_pivot_gold, &h_analysis_pivot_2);
        unit_check_general(1, 1, 1, &h_solve_pivot_gold, &h_solve_pivot_1);
        unit_check_general(1, 1, 1, &h_solve_pivot_gold, &h_solve_pivot_2);

        // Check solution matrix if no pivot has been found
        if(h_analysis_pivot_gold == -1 && h_solve_pivot_gold == -1)
        {
            unit_check_near(1, m * nrhs, 1, hB_gold.data(), hB_1.data());
            unit_check_near(1, m * nrhs, 1, hB_gold.data(), hB_2.data());
        }
    }
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSRSM2_HPP
