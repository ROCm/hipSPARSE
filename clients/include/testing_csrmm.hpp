/* ************************************************************************
 * Copyright (C) 2018-2019 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_CSRMM_HPP
#define TESTING_CSRMM_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include "hipsparse_arguments.hpp"

#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_csrmm_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                  N         = 100;
    int                  M         = 100;
    int                  K         = 100;
    int                  ldb       = 100;
    int                  ldc       = 100;
    int                  nnz       = 100;
    int                  safe_size = 100;
    T                    alpha     = 0.6;
    T                    beta      = 0.2;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseStatus_t    status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dC_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    int* dptr         = (int*)dptr_managed.get();
    int* dcol         = (int*)dcol_managed.get();
    T*   dval         = (T*)dval_managed.get();
    T*   dB           = (T*)dB_managed.get();
    T*   dC           = (T*)dC_managed.get();

    // testing for M = -1
    {
        status = hipsparseXcsrmm2(handle,
                                  transA,
                                  transB,
                                  -1,
                                  N,
                                  K,
                                  nnz,
                                  &alpha,
                                  descr,
                                  dval,
                                  dptr,
                                  dcol,
                                  dB,
                                  ldb,
                                  &beta,
                                  dC,
                                  ldc);
        verify_hipsparse_status_invalid_size(status, "Error: M < 0");
    }

    // testing for N = -1
    {
        status = hipsparseXcsrmm2(handle,
                                  transA,
                                  transB,
                                  M,
                                  -1,
                                  K,
                                  nnz,
                                  &alpha,
                                  descr,
                                  dval,
                                  dptr,
                                  dcol,
                                  dB,
                                  ldb,
                                  &beta,
                                  dC,
                                  ldc);
        verify_hipsparse_status_invalid_size(status, "Error: N < 0");
    }

    // testing for K = -1
    {
        status = hipsparseXcsrmm2(handle,
                                  transA,
                                  transB,
                                  M,
                                  N,
                                  -1,
                                  nnz,
                                  &alpha,
                                  descr,
                                  dval,
                                  dptr,
                                  dcol,
                                  dB,
                                  ldb,
                                  &beta,
                                  dC,
                                  ldc);
        verify_hipsparse_status_invalid_size(status, "Error: K < 0");
    }

    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXcsrmm2(handle,
                                  transA,
                                  transB,
                                  M,
                                  N,
                                  K,
                                  nnz,
                                  &alpha,
                                  descr,
                                  dval,
                                  dptr_null,
                                  dcol,
                                  dB,
                                  ldb,
                                  &beta,
                                  dC,
                                  ldc);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXcsrmm2(handle,
                                  transA,
                                  transB,
                                  M,
                                  N,
                                  K,
                                  nnz,
                                  &alpha,
                                  descr,
                                  dval,
                                  dptr,
                                  dcol_null,
                                  dB,
                                  ldb,
                                  &beta,
                                  dC,
                                  ldc);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXcsrmm2(handle,
                                  transA,
                                  transB,
                                  M,
                                  N,
                                  K,
                                  nnz,
                                  &alpha,
                                  descr,
                                  dval_null,
                                  dptr,
                                  dcol,
                                  dB,
                                  ldb,
                                  &beta,
                                  dC,
                                  ldc);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == dB)
    {
        T* dB_null = nullptr;

        status = hipsparseXcsrmm2(handle,
                                  transA,
                                  transB,
                                  M,
                                  N,
                                  K,
                                  nnz,
                                  &alpha,
                                  descr,
                                  dval,
                                  dptr,
                                  dcol,
                                  dB_null,
                                  ldb,
                                  &beta,
                                  dC,
                                  ldc);
        verify_hipsparse_status_invalid_pointer(status, "Error: dB is nullptr");
    }
    // testing for(nullptr == dC)
    {
        T* dC_null = nullptr;

        status = hipsparseXcsrmm2(handle,
                                  transA,
                                  transB,
                                  M,
                                  N,
                                  K,
                                  nnz,
                                  &alpha,
                                  descr,
                                  dval,
                                  dptr,
                                  dcol,
                                  dB,
                                  ldb,
                                  &beta,
                                  dC_null,
                                  ldc);
        verify_hipsparse_status_invalid_pointer(status, "Error: dC is nullptr");
    }
    // testing for(nullptr == d_alpha)
    {
        T* d_alpha_null = nullptr;

        status = hipsparseXcsrmm2(handle,
                                  transA,
                                  transB,
                                  M,
                                  N,
                                  K,
                                  nnz,
                                  d_alpha_null,
                                  descr,
                                  dval,
                                  dptr,
                                  dcol,
                                  dB,
                                  ldb,
                                  &beta,
                                  dC,
                                  ldc);
        verify_hipsparse_status_invalid_pointer(status, "Error: alpha is nullptr");
    }
    // testing for(nullptr == d_beta)
    {
        T* d_beta_null = nullptr;

        status = hipsparseXcsrmm2(handle,
                                  transA,
                                  transB,
                                  M,
                                  N,
                                  K,
                                  nnz,
                                  &alpha,
                                  descr,
                                  dval,
                                  dptr,
                                  dcol,
                                  dB,
                                  ldb,
                                  d_beta_null,
                                  dC,
                                  ldc);
        verify_hipsparse_status_invalid_pointer(status, "Error: beta is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXcsrmm2(handle,
                                  transA,
                                  transB,
                                  M,
                                  N,
                                  K,
                                  nnz,
                                  &alpha,
                                  descr_null,
                                  dval,
                                  dptr,
                                  dcol,
                                  dB,
                                  ldb,
                                  &beta,
                                  dC,
                                  ldc);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrmm2(handle_null,
                                  transA,
                                  transB,
                                  M,
                                  N,
                                  K,
                                  nnz,
                                  &alpha,
                                  descr,
                                  dval,
                                  dptr,
                                  dcol,
                                  dB,
                                  ldb,
                                  &beta,
                                  dC,
                                  ldc);
        verify_hipsparse_status_invalid_handle(status);
    }
#endif
}

template <typename T>
hipsparseStatus_t testing_csrmm(Arguments argus)
{
    int                  M        = argus.M;
    int                  N        = argus.N;
    int                  K        = argus.K;
    T                    h_alpha  = make_DataType<T>(argus.alpha);
    T                    h_beta   = make_DataType<T>(argus.beta);
    hipsparseOperation_t transA   = argus.transA;
    hipsparseOperation_t transB   = argus.transB;
    hipsparseIndexBase_t idx_base = argus.baseA;
    std::string          filename = argus.filename;

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr(new descr_struct);
    hipsparseMatDescr_t           descr = test_descr->descr;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptrA;
    std::vector<int> hcsr_col_indA;
    std::vector<T>   hcsr_valA;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_csr_matrix(filename, M, K, nnz, hcsr_row_ptrA, hcsr_col_indA, hcsr_valA, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // Some matrix properties
    int A_m = M;
    int B_m = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE)
                  ? (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K : M)
                  : N;
    int B_n = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE)
                  ? N
                  : (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K : M);

    int C_m   = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? M : K);
    int C_n   = N;
    int ldb   = B_m;
    int ldc   = C_m;
    int nrowB = ldb;
    int ncolB = B_n;
    int nrowC = ldc;
    int ncolC = C_n;
    int Bnnz  = nrowB * ncolB;
    int Cnnz  = nrowC * ncolC;

    // Host structures - Dense matrix B and C
    std::vector<T> hB(Bnnz);
    std::vector<T> hC_1(Cnnz);
    std::vector<T> hC_2(Cnnz);
    std::vector<T> hC_gold(Cnnz);

    hipsparseInit<T>(hB, nrowB, ncolB);
    hipsparseInit<T>(hC_1, nrowC, ncolC);

    // copy vector is easy in STL; hC_gold = hC_1: save a copy in hy_gold which will be output of
    // CPU
    hC_gold = hC_1;
    hC_2    = hC_1;

    // allocate memory on device
    auto dcsr_row_ptrA_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (A_m + 1)), device_free};
    auto dcsr_col_indA_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_valA_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dB_managed        = hipsparse_unique_ptr{device_malloc(sizeof(T) * Bnnz), device_free};
    auto dC_1_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * Cnnz), device_free};
    auto dC_2_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * Cnnz), device_free};
    auto d_alpha_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int* dcsr_row_ptrA = (int*)dcsr_row_ptrA_managed.get();
    int* dcsr_col_indA = (int*)dcsr_col_indA_managed.get();
    T*   dcsr_valA     = (T*)dcsr_valA_managed.get();
    T*   dB            = (T*)dB_managed.get();
    T*   dC_1          = (T*)dC_1_managed.get();
    T*   dC_2          = (T*)dC_2_managed.get();
    T*   d_alpha       = (T*)d_alpha_managed.get();
    T*   d_beta        = (T*)d_beta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptrA, hcsr_row_ptrA.data(), sizeof(int) * (A_m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_indA, hcsr_col_indA.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_valA, hcsr_valA.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * Bnnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1.data(), sizeof(T) * Cnnz, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2.data(), sizeof(T) * Cnnz, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        // ROCSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrmm2(handle,
                                               transA,
                                               transB,
                                               M,
                                               N,
                                               K,
                                               nnz,
                                               &h_alpha,
                                               descr,
                                               dcsr_valA,
                                               dcsr_row_ptrA,
                                               dcsr_col_indA,
                                               dB,
                                               ldb,
                                               &h_beta,
                                               dC_1,
                                               ldc));

        // ROCSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrmm2(handle,
                                               transA,
                                               transB,
                                               M,
                                               N,
                                               K,
                                               nnz,
                                               d_alpha,
                                               descr,
                                               dcsr_valA,
                                               dcsr_row_ptrA,
                                               dcsr_col_indA,
                                               dB,
                                               ldb,
                                               d_beta,
                                               dC_2,
                                               ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC_1, sizeof(T) * Cnnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC_2, sizeof(T) * Cnnz, hipMemcpyDeviceToHost));

        // CPU
        host_csrmm(M,
                   N,
                   K,
                   transA,
                   transB,
                   h_alpha,
                   hcsr_row_ptrA.data(),
                   hcsr_col_indA.data(),
                   hcsr_valA.data(),
                   hB.data(),
                   ldb,
                   h_beta,
                   hC_gold.data(),
                   ldc,
                   HIPSPARSE_ORDER_COL,
                   idx_base,
                   false);

        unit_check_near(nrowC, ncolC, ldc, hC_gold.data(), hC_1.data());
        unit_check_near(nrowC, ncolC, ldc, hC_gold.data(), hC_2.data());
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSRMM_HPP
