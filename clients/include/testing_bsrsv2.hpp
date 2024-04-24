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
#ifndef TESTING_BSRSV2_HPP
#define TESTING_BSRSV2_HPP

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
void testing_bsrsv2_bad_arg(void)
{
    /*
#ifdef __HIP_PLATFORM_NVIDIA__
    // do not test for bad args
    return;
#endif
    int                    m         = 100;
    int                    nnz       = 100;
    int                    safe_size = 100;
    T                      h_alpha   = 0.6;
    hipsparseOperation_t   transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseSolvePolicy_t policy    = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
    hipsparseStatus_t      status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<bsrsv2_struct> unique_ptr_bsrsv2_info(new bsrsv2_struct);
    bsrsv2Info_t                   info = unique_ptr_bsrsv2_info->info;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dy_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*  dptr    = (int*)dptr_managed.get();
    int*  dcol    = (int*)dcol_managed.get();
    T*    dval    = (T*)dval_managed.get();
    T*    dx      = (T*)dx_managed.get();
    T*    dy      = (T*)dy_managed.get();
    void* dbuffer = (void*)dbuffer_managed.get();

    if(!dval || !dptr || !dcol || !dx || !dy || !dbuffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing hipsparseXbsrsv2_bufferSize
    int size;

    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXbsrsv2_bufferSize(
            handle, transA, m, nnz, descr, dval, dptr_null, dcol, info, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXbsrsv2_bufferSize(
            handle, transA, m, nnz, descr, dval, dptr, dcol_null, info, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXbsrsv2_bufferSize(
            handle, transA, m, nnz, descr, dval_null, dptr, dcol, info, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == buffer_size)
    {
        int* size_null = nullptr;

        status = hipsparseXbsrsv2_bufferSize(
            handle, transA, m, nnz, descr, dval, dptr, dcol, info, size_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: size is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXbsrsv2_bufferSize(
            handle, transA, m, nnz, descr_null, dval, dptr, dcol, info, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        bsrsv2Info_t info_null = nullptr;

        status = hipsparseXbsrsv2_bufferSize(
            handle, transA, m, nnz, descr, dval, dptr, dcol, info_null, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXbsrsv2_bufferSize(
            handle_null, transA, m, nnz, descr, dval, dptr, dcol, info, &size);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing hipsparseXbsrsv2_analysis

    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXbsrsv2_analysis(
            handle, transA, m, nnz, descr, dval, dptr_null, dcol, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXbsrsv2_analysis(
            handle, transA, m, nnz, descr, dval, dptr, dcol_null, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXbsrsv2_analysis(
            handle, transA, m, nnz, descr, dval_null, dptr, dcol, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        void* dbuffer_null = nullptr;

        status = hipsparseXbsrsv2_analysis(
            handle, transA, m, nnz, descr, dval, dptr, dcol, info, policy, dbuffer_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXbsrsv2_analysis(
            handle, transA, m, nnz, descr_null, dval, dptr, dcol, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        bsrsv2Info_t info_null = nullptr;

        status = hipsparseXbsrsv2_analysis(
            handle, transA, m, nnz, descr, dval, dptr, dcol, info_null, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXbsrsv2_analysis(
            handle_null, transA, m, nnz, descr, dval, dptr, dcol, info, policy, dbuffer);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing rocsparse_bsrsv2

    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXbsrsv2_solve(handle,
                                        transA,
                                        m,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval,
                                        dptr_null,
                                        dcol,
                                        info,
                                        dx,
                                        dy,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXbsrsv2_solve(handle,
                                        transA,
                                        m,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval,
                                        dptr,
                                        dcol_null,
                                        info,
                                        dx,
                                        dy,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXbsrsv2_solve(handle,
                                        transA,
                                        m,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval_null,
                                        dptr,
                                        dcol,
                                        info,
                                        dx,
                                        dy,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == dx)
    {
        T* dx_null = nullptr;

        status = hipsparseXbsrsv2_solve(handle,
                                        transA,
                                        m,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval,
                                        dptr,
                                        dcol,
                                        info,
                                        dx_null,
                                        dy,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dx is nullptr");
    }
    // testing for(nullptr == dy)
    {
        T* dy_null = nullptr;

        status = hipsparseXbsrsv2_solve(handle,
                                        transA,
                                        m,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval,
                                        dptr,
                                        dcol,
                                        info,
                                        dx,
                                        dy_null,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dy is nullptr");
    }
    // testing for(nullptr == d_alpha)
    {
        T* d_alpha_null = nullptr;

        status = hipsparseXbsrsv2_solve(handle,
                                        transA,
                                        m,
                                        nnz,
                                        d_alpha_null,
                                        descr,
                                        dval,
                                        dptr,
                                        dcol,
                                        info,
                                        dx,
                                        dy,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: alpha is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        void* dbuffer_null = nullptr;

        status = hipsparseXbsrsv2_solve(handle,
                                        transA,
                                        m,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval,
                                        dptr,
                                        dcol,
                                        info,
                                        dx,
                                        dy,
                                        policy,
                                        dbuffer_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXbsrsv2_solve(handle,
                                        transA,
                                        m,
                                        nnz,
                                        &h_alpha,
                                        descr_null,
                                        dval,
                                        dptr,
                                        dcol,
                                        info,
                                        dx,
                                        dy,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        bsrsv2Info_t info_null = nullptr;

        status = hipsparseXbsrsv2_solve(handle,
                                        transA,
                                        m,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval,
                                        dptr,
                                        dcol,
                                        info_null,
                                        dx,
                                        dy,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXbsrsv2_solve(handle_null,
                                        transA,
                                        m,
                                        nnz,
                                        &h_alpha,
                                        descr,
                                        dval,
                                        dptr,
                                        dcol,
                                        info,
                                        dx,
                                        dy,
                                        policy,
                                        dbuffer);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing hipsparseXbsrsv2_zeroPivot
    int position;

    // testing for(nullptr == position)
    {
        int* position_null = nullptr;

        status = hipsparseXbsrsv2_zeroPivot(handle, info, position_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: position is nullptr");
    }
    // testing for(nullptr == info)
    {
        bsrsv2Info_t info_null = nullptr;

        status = hipsparseXbsrsv2_zeroPivot(handle, info_null, &position);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXbsrsv2_zeroPivot(handle_null, info, &position);
        verify_hipsparse_status_invalid_handle(status);
    }
*/
}

template <typename T>
hipsparseStatus_t testing_bsrsv2(Arguments argus)
{
    int                    safe_size = 100;
    int                    m         = argus.M;
    int                    block_dim = argus.block_dim;
    hipsparseIndexBase_t   idx_base  = argus.idx_base;
    hipsparseDirection_t   dir       = argus.dirA;
    hipsparseOperation_t   trans     = argus.transA;
    hipsparseDiagType_t    diag_type = argus.diag_type;
    hipsparseFillMode_t    fill_mode = argus.fill_mode;
    hipsparseSolvePolicy_t policy    = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
    T                      h_alpha   = make_DataType<T>(argus.alpha);
    std::string            binfile   = "";
    std::string            filename  = "";
    hipsparseStatus_t      status;
    int                    size;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    if(m == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        m       = safe_size;
    }

    if(argus.timing == 1)
    {
        filename = argus.filename;
    }

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr(new descr_struct);
    hipsparseMatDescr_t           descr = test_descr->descr;

    std::unique_ptr<bsrsv2_struct> unique_ptr_bsrsv2_info(new bsrsv2_struct);
    bsrsv2Info_t                   info = unique_ptr_bsrsv2_info->info;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    // Set matrix diag type
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatDiagType(descr, diag_type));

    // Set matrix fill mode
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatFillMode(descr, fill_mode));

    int mb = (block_dim < 1) ? -1 : (m + block_dim - 1) / block_dim;

//     // Argument sanity check before allocating invalid memory
//     if(mb <= 0 || m <= 0 || block_dim <= 0)
//     {
// #ifdef __HIP_PLATFORM_NVIDIA__
//         // Do not test args in cusparse
//         return HIPSPARSE_STATUS_SUCCESS;
// #endif

//         auto dptr_managed
//             = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
//         auto dcol_managed
//             = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
//         auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
//         auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
//         auto dy_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
//         auto buffer_managed
//             = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

//         int*  dptr   = (int*)dptr_managed.get();
//         int*  dcol   = (int*)dcol_managed.get();
//         T*    dval   = (T*)dval_managed.get();
//         T*    dx     = (T*)dx_managed.get();
//         T*    dy     = (T*)dy_managed.get();
//         void* buffer = (void*)buffer_managed.get();

//         if(!dval || !dptr || !dcol || !dx || !dy || !buffer)
//         {
//             verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
//                                             "!dptr || !dcol || !dval || "
//                                             "!dx || !dy || !buffer");
//             return HIPSPARSE_STATUS_ALLOC_FAILED;
//         }

//         // Test hipsparseXbsrsv2_bufferSize
//         {
//             size_t lsize;
//             status = hipsparseXbsrsv2_bufferSizeExt(handle,
//                                                     dir,
//                                                     trans,
//                                                     mb,
//                                                     safe_size,
//                                                     descr,
//                                                     dval,
//                                                     dptr,
//                                                     dcol,
//                                                     block_dim,
//                                                     info,
//                                                     &lsize);
//         }

//         if(mb < 0 || block_dim < 0)
//         {
//             verify_hipsparse_status_invalid_size(status, "Error: mb < 0 || block_dim < 0");
//         }
//         else
//         {
//             verify_hipsparse_status_success(status, "mb >= 0 && block_dim >= 0");
//         }

//         // Test hipsparseXbsrsv2_analysis
//         status = hipsparseXbsrsv2_analysis(handle,
//                                            dir,
//                                            trans,
//                                            mb,
//                                            safe_size,
//                                            descr,
//                                            dval,
//                                            dptr,
//                                            dcol,
//                                            block_dim,
//                                            info,
//                                            policy,
//                                            buffer);

//         if(mb < 0 || block_dim < 0)
//         {
//             verify_hipsparse_status_invalid_size(status, "Error: mb < 0 || block_dim < 0");
//         }
//         else
//         {
//             verify_hipsparse_status_success(status, "mb >= 0 && block_dim >= 0");
//         }

//         // Test hipsparseXbsrsv2_solve
//         status = hipsparseXbsrsv2_solve(handle,
//                                         dir,
//                                         trans,
//                                         mb,
//                                         safe_size,
//                                         &h_alpha,
//                                         descr,
//                                         dval,
//                                         dptr,
//                                         dcol,
//                                         block_dim,
//                                         info,
//                                         dx,
//                                         dy,
//                                         policy,
//                                         buffer);

//         if(mb < 0 || block_dim < 0)
//         {
//             verify_hipsparse_status_invalid_size(status, "Error: mb < 0 || block_dim < 0");
//         }
//         else
//         {
//             verify_hipsparse_status_success(status, "mb >= 0 && block_dim >= 0");
//         }

//         // Test hipsparseXbsrsv2_zeroPivot
//         int zero_pivot;
//         CHECK_HIPSPARSE_ERROR(hipsparseXbsrsv2_zeroPivot(handle, info, &zero_pivot));

//         // Zero pivot should be -1
//         int res = -1;
//         unit_check_general(1, 1, 1, &res, &zero_pivot);

//         return HIPSPARSE_STATUS_SUCCESS;
//     }

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;
    int              nnz;

    // Initial Data on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        int n;
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
        m   = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base);
        nnz = hcsr_row_ptr[m];
    }
    else
    {
        std::vector<int> hcoo_row_ind;

        if(filename != "")
        {
            int n;
            if(read_mtx_matrix(
                   filename.c_str(), m, n, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, idx_base)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            double scale = 0.02;
            if(m > 1000)
            {
                scale = 2.0 / m;
            }
            nnz = m * scale * m;

            gen_matrix_coo(m, m, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, idx_base);
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

    mb = (m + block_dim - 1) / block_dim;

    std::vector<T> hx(mb * block_dim);
    std::vector<T> hy_1(mb * block_dim);
    std::vector<T> hy_2(mb * block_dim);
    std::vector<T> hy_gold(mb * block_dim);

    hipsparseInit<T>(hx, 1, mb * block_dim);

    // Allocate memory on device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};
    auto dx_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * mb * block_dim), device_free};
    auto dy_1_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * mb * block_dim), device_free};
    auto dy_2_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * mb * block_dim), device_free};
    auto d_alpha_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_position_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val     = (T*)dcsr_val_managed.get();
    int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();
    T*   dx           = (T*)dx_managed.get();
    T*   dy_1         = (T*)dy_1_managed.get();
    T*   dy_2         = (T*)dy_2_managed.get();
    T*   d_alpha      = (T*)d_alpha_managed.get();
    int* d_position   = (int*)d_position_managed.get();

    if(!dcsr_val || !dcsr_row_ptr || !dcsr_col_ind || !dbsr_row_ptr || !dx || !dy_1 || !dy_2
       || !d_alpha || !d_position)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dval || !dptr || !dcol || !dx || "
                                        "!dy_1 || !dy_2 || !d_alpha || !d_position");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // Convert to BSR
    int nnzb;
    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsrNnz(handle,
                                               dir,
                                               m,
                                               m,
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

    if(!dbsr_val || !dbsr_col_ind)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dbsr_val || !dbsr_col_ind");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsr(handle,
                                            dir,
                                            m,
                                            m,
                                            descr,
                                            dcsr_val,
                                            dcsr_row_ptr,
                                            dcsr_col_ind,
                                            block_dim,
                                            descr,
                                            dbsr_val,
                                            dbsr_row_ptr,
                                            dbsr_col_ind));

    // Obtain bsrsv2 buffer size
    CHECK_HIPSPARSE_ERROR(hipsparseXbsrsv2_bufferSize(handle,
                                                      dir,
                                                      trans,
                                                      mb,
                                                      nnzb,
                                                      descr,
                                                      dbsr_val,
                                                      dbsr_row_ptr,
                                                      dbsr_col_ind,
                                                      block_dim,
                                                      info,
                                                      &size));

    // Allocate buffer on the device
    auto dbuffer_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * size), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    if(!dbuffer)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED, "!dbuffer");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // bsrsv2 analysis
    CHECK_HIPSPARSE_ERROR(hipsparseXbsrsv2_analysis(handle,
                                                    dir,
                                                    trans,
                                                    mb,
                                                    nnzb,
                                                    descr,
                                                    dbsr_val,
                                                    dbsr_row_ptr,
                                                    dbsr_col_ind,
                                                    block_dim,
                                                    info,
                                                    policy,
                                                    dbuffer));

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * m, hipMemcpyHostToDevice));

        // ROCSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXbsrsv2_solve(handle,
                                                     dir,
                                                     trans,
                                                     mb,
                                                     nnzb,
                                                     &h_alpha,
                                                     descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     block_dim,
                                                     info,
                                                     dx,
                                                     dy_1,
                                                     policy,
                                                     dbuffer));

        int               hposition_1;
        hipsparseStatus_t pivot_status_1;
        pivot_status_1 = hipsparseXbsrsv2_zeroPivot(handle, info, &hposition_1);

        // ROCSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXbsrsv2_solve(handle,
                                                     dir,
                                                     trans,
                                                     mb,
                                                     nnzb,
                                                     d_alpha,
                                                     descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     block_dim,
                                                     info,
                                                     dx,
                                                     dy_2,
                                                     policy,
                                                     dbuffer));

        hipsparseStatus_t pivot_status_2;
        pivot_status_2 = hipsparseXbsrsv2_zeroPivot(handle, info, d_position);

        // Copy output from device to CPU
        int hposition_2;
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * m, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * m, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(&hposition_2, d_position, sizeof(int), hipMemcpyDeviceToHost));

        // Host bsrsv2
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

        int struct_position_gold;
        int position_gold;

        bsrsv(trans,
              dir,
              mb,
              nnzb,
              h_alpha,
              hbsr_row_ptr.data(),
              hbsr_col_ind.data(),
              hbsr_val.data(),
              block_dim,
              hx.data(),
              hy_gold.data(),
              diag_type,
              fill_mode,
              idx_base,
              &struct_position_gold,
              &position_gold);

        unit_check_general(1, 1, 1, &position_gold, &hposition_1);
        unit_check_general(1, 1, 1, &position_gold, &hposition_2);

        if(hposition_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(pivot_status_1,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
            return HIPSPARSE_STATUS_SUCCESS;
        }

        if(hposition_2 != -1)
        {
            verify_hipsparse_status_zero_pivot(pivot_status_2,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
            return HIPSPARSE_STATUS_SUCCESS;
        }

        unit_check_near(1, m, 1, hy_gold.data(), hy_1.data());
        unit_check_near(1, m, 1, hy_gold.data(), hy_2.data());
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_BSRSV2_HPP
