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
#ifndef TESTING_BSRILU02_HPP
#define TESTING_BSRILU02_HPP

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
void testing_bsrilu02_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                    mb        = 100;
    int                    nnzb      = 100;
    int                    block_dim = 4;
    int                    safe_size = 100;
    hipsparseDirection_t   dirA      = HIPSPARSE_DIRECTION_COLUMN;
    hipsparseSolvePolicy_t policy    = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
    hipsparseStatus_t      status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<bsrilu02_struct> unique_ptr_bsrilu02(new bsrilu02_struct);
    bsrilu02Info_t                   info = unique_ptr_bsrilu02->info;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};
    auto dboost_tol_managed = hipsparse_unique_ptr{device_malloc(sizeof(double)), device_free};
    auto dboost_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int*    dptr       = (int*)dptr_managed.get();
    int*    dcol       = (int*)dcol_managed.get();
    T*      dval       = (T*)dval_managed.get();
    void*   dbuffer    = (void*)dbuffer_managed.get();
    double* dboost_tol = (double*)dboost_tol_managed.get();
    T*      dboost_val = (T*)dboost_val_managed.get();

    if(!dval || !dptr || !dcol || !dbuffer || !dboost_tol || !dboost_val)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing hipsparseXbsrilu02_bufferSize
    int size;

    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXbsrilu02_bufferSize(
            handle, dirA, mb, nnzb, descr, dval, dptr_null, dcol, block_dim, info, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXbsrilu02_bufferSize(
            handle, dirA, mb, nnzb, descr, dval, dptr, dcol_null, block_dim, info, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXbsrilu02_bufferSize(
            handle, dirA, mb, nnzb, descr, dval_null, dptr, dcol, block_dim, info, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == buffer_size)
    {
        int* size_null = nullptr;

        status = hipsparseXbsrilu02_bufferSize(
            handle, dirA, mb, nnzb, descr, dval, dptr, dcol, block_dim, info, size_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: size is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXbsrilu02_bufferSize(
            handle, dirA, mb, nnzb, descr_null, dval, dptr, dcol, block_dim, info, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        bsrilu02Info_t info_null = nullptr;

        status = hipsparseXbsrilu02_bufferSize(
            handle, dirA, mb, nnzb, descr, dval, dptr, dcol, block_dim, info_null, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXbsrilu02_bufferSize(
            handle_null, dirA, mb, nnzb, descr, dval, dptr, dcol, block_dim, info, &size);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing hipsparseXbsrilu02_numericBoost

    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXbsrilu02_numericBoost(handle_null, info, 1, dboost_tol, dboost_val);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing for(nullptr == info)
    {
        bsrilu02Info_t info_null = nullptr;

        status = hipsparseXbsrilu02_numericBoost(handle, info_null, 1, dboost_tol, dboost_val);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }

    // testing for(nullptr == dboost_tol)
    {
        double* boost_tol_null = nullptr;

        status = hipsparseXbsrilu02_numericBoost(handle, info, 1, boost_tol_null, dboost_val);
        verify_hipsparse_status_invalid_pointer(status, "Error: boost_tol is nullptr");
    }

    // testing for(nullptr == dboost_val)
    {
        T* boost_val_null = nullptr;

        status = hipsparseXbsrilu02_numericBoost(handle, info, 1, dboost_tol, boost_val_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: boost_val is nullptr");
    }

    // testing hipsparseXbsrilu02_analysis

    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXbsrilu02_analysis(
            handle, dirA, mb, nnzb, descr, dval, dptr_null, dcol, block_dim, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXbsrilu02_analysis(
            handle, dirA, mb, nnzb, descr, dval, dptr, dcol_null, block_dim, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXbsrilu02_analysis(
            handle, dirA, mb, nnzb, descr, dval_null, dptr, dcol, block_dim, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        void* dbuffer_null = nullptr;

        status = hipsparseXbsrilu02_analysis(
            handle, dirA, mb, nnzb, descr, dval, dptr, dcol, block_dim, info, policy, dbuffer_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXbsrilu02_analysis(
            handle, dirA, mb, nnzb, descr_null, dval, dptr, dcol, block_dim, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        bsrilu02Info_t info_null = nullptr;

        status = hipsparseXbsrilu02_analysis(
            handle, dirA, mb, nnzb, descr, dval, dptr, dcol, block_dim, info_null, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXbsrilu02_analysis(
            handle_null, dirA, mb, nnzb, descr, dval, dptr, dcol, block_dim, info, policy, dbuffer);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing hipsparseXbsrilu02

    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXbsrilu02(
            handle, dirA, mb, nnzb, descr, dval, dptr_null, dcol, block_dim, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXbsrilu02(
            handle, dirA, mb, nnzb, descr, dval, dptr, dcol_null, block_dim, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXbsrilu02(
            handle, dirA, mb, nnzb, descr, dval_null, dptr, dcol, block_dim, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        void* dbuffer_null = nullptr;

        status = hipsparseXbsrilu02(
            handle, dirA, mb, nnzb, descr, dval, dptr, dcol, block_dim, info, policy, dbuffer_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXbsrilu02(
            handle, dirA, mb, nnzb, descr_null, dval, dptr, dcol, block_dim, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        bsrilu02Info_t info_null = nullptr;

        status = hipsparseXbsrilu02(
            handle, dirA, mb, nnzb, descr, dval, dptr, dcol, block_dim, info_null, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXbsrilu02(
            handle_null, dirA, mb, nnzb, descr, dval, dptr, dcol, block_dim, info, policy, dbuffer);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing hipsparseXbsrilu02_zeroPivot
    int position;

    // testing for(nullptr == position)
    {
        int* position_null = nullptr;

        status = hipsparseXbsrilu02_zeroPivot(handle, info, position_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: position is nullptr");
    }
    // testing for(nullptr == info)
    {
        bsrilu02Info_t info_null = nullptr;

        status = hipsparseXbsrilu02_zeroPivot(handle, info_null, &position);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXbsrilu02_zeroPivot(handle_null, info, &position);
        verify_hipsparse_status_invalid_handle(status);
    }
#endif
}

template <typename T>
hipsparseStatus_t testing_bsrilu02(Arguments argus)
{
    int                    m         = argus.M;
    int                    block_dim = argus.block_dim;
    int                    boost     = argus.numericboost;
    double                 boost_tol = argus.boosttol;
    T                      boost_val = make_DataType<T>(argus.boostval, argus.boostvali);
    hipsparseDirection_t   dir       = argus.dirA;
    hipsparseIndexBase_t   idx_base  = argus.idx_base;
    hipsparseSolvePolicy_t policy    = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
    std::string            filename  = argus.filename;

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr(new descr_struct);
    hipsparseMatDescr_t           descr = test_descr->descr;

    std::unique_ptr<bsrilu02_struct> unique_ptr_bsrilu02(new bsrilu02_struct);
    bsrilu02Info_t                   info = unique_ptr_bsrilu02->info;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    if(m == 0)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        // cusparse does not support m == 0 for csr2bsr
        return HIPSPARSE_STATUS_SUCCESS;
#endif
    }

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

    // m can be modifed if we read in a matrix from a file
    int mb = (m + block_dim - 1) / block_dim;

    // allocate memory on device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};
    auto boost_tol_managed = hipsparse_unique_ptr{device_malloc(sizeof(double)), device_free};
    auto boost_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int*    dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int*    dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*      dcsr_val     = (T*)dcsr_val_managed.get();
    int*    dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();
    double* dboost_tol   = (double*)boost_tol_managed.get();
    T*      dboost_val   = (T*)boost_val_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

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
    auto dbsr_val_1_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(T) * nnzb * block_dim * block_dim), device_free};
    auto dbsr_val_2_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(T) * nnzb * block_dim * block_dim), device_free};
    auto d_analysis_pivot_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
    auto d_solve_pivot_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};

    int* dbsr_col_ind       = (int*)dbsr_col_ind_managed.get();
    T*   dbsr_val_1         = (T*)dbsr_val_1_managed.get();
    T*   dbsr_val_2         = (T*)dbsr_val_2_managed.get();
    int* d_analysis_pivot_2 = (int*)d_analysis_pivot_2_managed.get();
    int* d_solve_pivot_2    = (int*)d_solve_pivot_2_managed.get();

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
                                            dbsr_val_1,
                                            dbsr_row_ptr,
                                            dbsr_col_ind));

    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_val_2, dbsr_val_1, sizeof(T) * nnzb * block_dim * block_dim, hipMemcpyDeviceToDevice));

    // Host BSR matrix
    std::vector<int> hbsr_row_ptr(mb + 1);
    std::vector<int> hbsr_col_ind(nnzb);
    std::vector<T>   hbsr_val(nnzb * block_dim * block_dim);

    // Copy device BSR matrix to host
    CHECK_HIP_ERROR(hipMemcpy(
        hbsr_row_ptr.data(), dbsr_row_ptr, sizeof(int) * (mb + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hbsr_col_ind.data(), dbsr_col_ind, sizeof(int) * nnzb, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hbsr_val.data(),
                              dbsr_val_1,
                              sizeof(T) * nnzb * block_dim * block_dim,
                              hipMemcpyDeviceToHost));

    // Obtain bsrilu02 buffer size
    int bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseXbsrilu02_bufferSize(handle,
                                                        dir,
                                                        mb,
                                                        nnzb,
                                                        descr,
                                                        dbsr_val_1,
                                                        dbsr_row_ptr,
                                                        dbsr_col_ind,
                                                        block_dim,
                                                        info,
                                                        &bufferSize));

    // Allocate buffer on the device
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * bufferSize), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    int h_analysis_pivot_gold;
    int h_analysis_pivot_1;
    int h_analysis_pivot_2;
    int h_solve_pivot_gold;
    int h_solve_pivot_1;
    int h_solve_pivot_2;

    if(argus.unit_check)
    {
        hipsparseStatus_t status_analysis_1;
        hipsparseStatus_t status_analysis_2;
        hipsparseStatus_t status_solve_1;
        hipsparseStatus_t status_solve_2;

        CHECK_HIP_ERROR(hipMemcpy(dboost_tol, &boost_tol, sizeof(double), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dboost_val, &boost_val, sizeof(T), hipMemcpyHostToDevice));

        // bsrilu02 analysis - host mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXbsrilu02_analysis(handle,
                                                          dir,
                                                          mb,
                                                          nnzb,
                                                          descr,
                                                          dbsr_val_1,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          block_dim,
                                                          info,
                                                          policy,
                                                          dbuffer));

        // Get pivot
        status_analysis_1 = hipsparseXbsrilu02_zeroPivot(handle, info, &h_analysis_pivot_1);
        if(h_analysis_pivot_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(status_analysis_1,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        }

        // bsrilu02 analysis - device mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXbsrilu02_analysis(handle,
                                                          dir,
                                                          mb,
                                                          nnzb,
                                                          descr,
                                                          dbsr_val_2,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          block_dim,
                                                          info,
                                                          policy,
                                                          dbuffer));

        // Get pivot
        status_analysis_2 = hipsparseXbsrilu02_zeroPivot(handle, info, d_analysis_pivot_2);
        if(h_analysis_pivot_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(status_analysis_2,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        }

        // bsrilu02 solve - host mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXbsrilu02_numericBoost(handle, info, boost, &boost_tol, &boost_val));
        CHECK_HIPSPARSE_ERROR(hipsparseXbsrilu02(handle,
                                                 dir,
                                                 mb,
                                                 nnzb,
                                                 descr,
                                                 dbsr_val_1,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 block_dim,
                                                 info,
                                                 policy,
                                                 dbuffer));

        // Get pivot
        status_solve_1 = hipsparseXbsrilu02_zeroPivot(handle, info, &h_solve_pivot_1);
        if(h_solve_pivot_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(status_solve_1,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        }

        // bsrilu02 solve - device mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXbsrilu02_numericBoost(handle, info, boost, dboost_tol, dboost_val));
        CHECK_HIPSPARSE_ERROR(hipsparseXbsrilu02(handle,
                                                 dir,
                                                 mb,
                                                 nnzb,
                                                 descr,
                                                 dbsr_val_2,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 block_dim,
                                                 info,
                                                 policy,
                                                 dbuffer));

        // Get pivot
        status_solve_2 = hipsparseXbsrilu02_zeroPivot(handle, info, d_solve_pivot_2);
        if(h_solve_pivot_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(status_solve_2,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        }

        // Copy output from device to CPU
        std::vector<T> result_1(block_dim * block_dim * nnzb);
        std::vector<T> result_2(block_dim * block_dim * nnzb);

        CHECK_HIP_ERROR(hipMemcpy(result_1.data(),
                                  dbsr_val_1,
                                  sizeof(T) * block_dim * block_dim * nnzb,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(result_2.data(),
                                  dbsr_val_2,
                                  sizeof(T) * block_dim * block_dim * nnzb,
                                  hipMemcpyDeviceToHost));

        CHECK_HIP_ERROR(
            hipMemcpy(&h_analysis_pivot_2, d_analysis_pivot_2, sizeof(int), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(&h_solve_pivot_2, d_solve_pivot_2, sizeof(int), hipMemcpyDeviceToHost));

        // Host bsrilu02
        int numerical_pivot;
        int structural_pivot;
        host_bsrilu02<T>(dir,
                         mb,
                         block_dim,
                         hbsr_row_ptr,
                         hbsr_col_ind,
                         hbsr_val,
                         idx_base,
                         &structural_pivot,
                         &numerical_pivot,
                         boost,
                         boost_tol,
                         boost_val);

        h_analysis_pivot_gold = structural_pivot;

        // Solve pivot gives the first numerical or structural non-invertible block
        if(structural_pivot == -1)
        {
            h_solve_pivot_gold = numerical_pivot;
        }
        else if(numerical_pivot == -1)
        {
            h_solve_pivot_gold = structural_pivot;
        }
        else
        {
            h_solve_pivot_gold = std::min(numerical_pivot, structural_pivot);
        }

        unit_check_general(1, 1, 1, &h_analysis_pivot_gold, &h_analysis_pivot_1);
        unit_check_general(1, 1, 1, &h_analysis_pivot_gold, &h_analysis_pivot_2);
        unit_check_general(1, 1, 1, &h_solve_pivot_gold, &h_solve_pivot_1);
        unit_check_general(1, 1, 1, &h_solve_pivot_gold, &h_solve_pivot_2);

        if(h_analysis_pivot_gold == -1 && h_solve_pivot_gold == -1)
        {
            unit_check_near(1, nnzb * block_dim * block_dim, 1, hbsr_val.data(), result_1.data());
            unit_check_near(1, nnzb * block_dim * block_dim, 1, hbsr_val.data(), result_2.data());
        }
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_BSRILU02_HPP
