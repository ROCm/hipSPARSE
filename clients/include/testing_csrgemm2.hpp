/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#ifndef TESTING_CSRGEMM2_HPP
#define TESTING_CSRGEMM2_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_csrgemm2_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVCC__
    // do not test for bad args
    return;
#endif
    int M         = 100;
    int N         = 100;
    int K         = 100;
    int nnz_A     = 100;
    int nnz_B     = 100;
    int safe_size = 100;

    T alpha = 1.0;
    T zero  = 0.0;

    hipsparseStatus_t status;
    size_t            size;
    int               nnz_C;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr_A(new descr_struct);
    hipsparseMatDescr_t           descr_A = unique_ptr_descr_A->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_B(new descr_struct);
    hipsparseMatDescr_t           descr_B = unique_ptr_descr_B->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_C(new descr_struct);
    hipsparseMatDescr_t           descr_C = unique_ptr_descr_C->descr;

    std::unique_ptr<csrgemm2_struct> unique_ptr_csrgemm2(new csrgemm2_struct);
    csrgemm2Info_t                   info = unique_ptr_csrgemm2->info;

    auto dAptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dAcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dAval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dBptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dBcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dBval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dCptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dCcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dCval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*  dAptr   = (int*)dAptr_managed.get();
    int*  dAcol   = (int*)dAcol_managed.get();
    T*    dAval   = (T*)dAval_managed.get();
    int*  dBptr   = (int*)dBptr_managed.get();
    int*  dBcol   = (int*)dBcol_managed.get();
    T*    dBval   = (T*)dBval_managed.get();
    int*  dCptr   = (int*)dCptr_managed.get();
    int*  dCcol   = (int*)dCcol_managed.get();
    T*    dCval   = (T*)dCval_managed.get();
    void* dbuffer = (void*)dbuffer_managed.get();

    if(!dAval || !dAptr || !dAcol || !dBval || !dBptr || !dBcol || !dCval || !dCptr || !dCcol
       || !dbuffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Scenario: alpha != 0 and beta == 0

    // testing hipsparseXcsrgemm2_bufferSizeExt

    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrgemm2_bufferSizeExt(handle_null,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  dAptr,
                                                  dAcol,
                                                  descr_B,
                                                  nnz_B,
                                                  dBptr,
                                                  dBcol,
                                                  (T*)nullptr,
                                                  (hipsparseMatDescr_t) nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  (int*)nullptr,
                                                  info,
                                                  &size);
        verify_hipsparse_status_invalid_handle(status);
    }
    // testing for(nullptr == alpha)
    {
        T* alpha_null = nullptr;

        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  K,
                                                  alpha_null,
                                                  descr_A,
                                                  nnz_A,
                                                  dAptr,
                                                  dAcol,
                                                  descr_B,
                                                  nnz_B,
                                                  dBptr,
                                                  dBcol,
                                                  (T*)nullptr,
                                                  (hipsparseMatDescr_t) nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  (int*)nullptr,
                                                  info,
                                                  &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: alpha is nullptr");
    }
    // testing for(nullptr == descr_A)
    {
        hipsparseMatDescr_t descr_A_null = nullptr;

        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  descr_A_null,
                                                  nnz_A,
                                                  dAptr,
                                                  dAcol,
                                                  descr_B,
                                                  nnz_B,
                                                  dBptr,
                                                  dBcol,
                                                  (T*)nullptr,
                                                  (hipsparseMatDescr_t) nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  (int*)nullptr,
                                                  info,
                                                  &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");
    }
    // testing for(nullptr == dAptr)
    {
        int* dAptr_null = nullptr;

        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  dAptr_null,
                                                  dAcol,
                                                  descr_B,
                                                  nnz_B,
                                                  dBptr,
                                                  dBcol,
                                                  (T*)nullptr,
                                                  (hipsparseMatDescr_t) nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  (int*)nullptr,
                                                  info,
                                                  &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dAptr is nullptr");
    }
    // testing for(nullptr == dAcol)
    {
        int* dAcol_null = nullptr;

        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  dAptr,
                                                  dAcol_null,
                                                  descr_B,
                                                  nnz_B,
                                                  dBptr,
                                                  dBcol,
                                                  (T*)nullptr,
                                                  (hipsparseMatDescr_t) nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  (int*)nullptr,
                                                  info,
                                                  &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dAcol is nullptr");
    }
    // testing for(nullptr == descr_B)
    {
        hipsparseMatDescr_t descr_B_null = nullptr;

        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  dAptr,
                                                  dAcol,
                                                  descr_B_null,
                                                  nnz_B,
                                                  dBptr,
                                                  dBcol,
                                                  (T*)nullptr,
                                                  (hipsparseMatDescr_t) nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  (int*)nullptr,
                                                  info,
                                                  &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_B is nullptr");
    }
    // testing for(nullptr == dBptr)
    {
        int* dBptr_null = nullptr;

        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  dAptr,
                                                  dAcol,
                                                  descr_B,
                                                  nnz_B,
                                                  dBptr_null,
                                                  dBcol,
                                                  (T*)nullptr,
                                                  (hipsparseMatDescr_t) nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  (int*)nullptr,
                                                  info,
                                                  &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dBptr is nullptr");
    }
    // testing for(nullptr == dBcol)
    {
        int* dBcol_null = nullptr;

        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  dAptr,
                                                  dAcol,
                                                  descr_B,
                                                  nnz_B,
                                                  dBptr,
                                                  dBcol_null,
                                                  (T*)nullptr,
                                                  (hipsparseMatDescr_t) nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  (int*)nullptr,
                                                  info,
                                                  &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dBcol is nullptr");
    }
    // testing for(nullptr == info)
    {
        csrgemm2Info_t info_null = nullptr;

        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  dAptr,
                                                  dAcol,
                                                  descr_B,
                                                  nnz_B,
                                                  dBptr,
                                                  dBcol,
                                                  (T*)nullptr,
                                                  (hipsparseMatDescr_t) nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  (int*)nullptr,
                                                  info_null,
                                                  &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == size)
    {
        size_t* size_null = nullptr;

        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  K,
                                                  &alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  dAptr,
                                                  dAcol,
                                                  descr_B,
                                                  nnz_B,
                                                  dBptr,
                                                  dBcol,
                                                  (T*)nullptr,
                                                  (hipsparseMatDescr_t) nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  (int*)nullptr,
                                                  info,
                                                  size_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: size is nullptr");
    }

    // testing hipsparseXcsrgemm2Nnz

    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrgemm2Nnz(handle_null,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_handle(status);
    }
    // testing for(nullptr == descr_A)
    {
        hipsparseMatDescr_t descr_A_null = nullptr;

        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       K,
                                       descr_A_null,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");
    }
    // testing for(nullptr == dAptr)
    {
        int* dAptr_null = nullptr;

        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr_null,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dAptr is nullptr");
    }
    // testing for(nullptr == dAcol)
    {
        int* dAcol_null = nullptr;

        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol_null,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dAcol is nullptr");
    }
    // testing for(nullptr == descr_B)
    {
        hipsparseMatDescr_t descr_B_null = nullptr;

        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B_null,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_B is nullptr");
    }
    // testing for(nullptr == dBptr)
    {
        int* dBptr_null = nullptr;

        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr_null,
                                       dBcol,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dBptr is nullptr");
    }
    // testing for(nullptr == dBcol)
    {
        int* dBcol_null = nullptr;

        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol_null,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dBcol is nullptr");
    }
    // testing for(nullptr == descr_C)
    {
        hipsparseMatDescr_t descr_C_null = nullptr;

        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C_null,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");
    }
    // testing for(nullptr == dCptr)
    {
        int* dCptr_null = nullptr;

        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C,
                                       dCptr_null,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dCptr is nullptr");
    }
    // testing for(nullptr == nnz_C)
    {
        int* nnz_C_null = nullptr;

        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C,
                                       dCptr,
                                       nnz_C_null,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: nnz_C is nullptr");
    }
    // testing for(nullptr == info)
    {
        csrgemm2Info_t info_null = nullptr;

        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info_null,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        void* dbuffer_null = nullptr;

        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }

    // testing hipsparseXcsrgemm2

    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrgemm2(handle_null,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_handle(status);
    }
    // testing for(nullptr == alpha)
    {
        T* alpha_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    alpha_null,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: alpha is nullptr");
    }
    // testing for(nullptr == descr_A)
    {
        hipsparseMatDescr_t descr_A_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A_null,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");
    }
    // testing for(nullptr == dAval)
    {
        T* dAval_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval_null,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dAval is nullptr");
    }
    // testing for(nullptr == dAptr)
    {
        int* dAptr_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr_null,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dAptr is nullptr");
    }
    // testing for(nullptr == dAcol)
    {
        int* dAcol_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol_null,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dAcol is nullptr");
    }
    // testing for(nullptr == descr_B)
    {
        hipsparseMatDescr_t descr_B_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B_null,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_B is nullptr");
    }
    // testing for(nullptr == dBval)
    {
        T* dBval_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval_null,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dBval is nullptr");
    }
    // testing for(nullptr == dBptr)
    {
        int* dBptr_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr_null,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dBptr is nullptr");
    }
    // testing for(nullptr == dBcol)
    {
        int* dBcol_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol_null,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dBcol is nullptr");
    }
    // testing for(nullptr == descr_C)
    {
        hipsparseMatDescr_t descr_C_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C_null,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");
    }
    // testing for(nullptr == dCval)
    {
        T* dCval_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval_null,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dCval is nullptr");
    }
    // testing for(nullptr == dCptr)
    {
        int* dCptr_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr_null,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dCptr is nullptr");
    }
    // testing for(nullptr == dCcol)
    {
        int* dCcol_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol_null,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dCcol is nullptr");
    }
    // testing for(nullptr == info)
    {
        csrgemm2Info_t info_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info_null,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        void* dbuffer_null = nullptr;

        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    (hipsparseMatDescr_t) nullptr,
                                    0,
                                    (T*)nullptr,
                                    (int*)nullptr,
                                    (int*)nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
}

template <typename T>
static int csrgemm2_nnz(int                  m,
                        int                  n,
                        int                  k,
                        const int*           csr_row_ptr_A,
                        const int*           csr_col_ind_A,
                        const int*           csr_row_ptr_B,
                        const int*           csr_col_ind_B,
                        T                    beta,
                        const int*           csr_row_ptr_D,
                        const int*           csr_col_ind_D,
                        int*                 csr_row_ptr_C,
                        hipsparseIndexBase_t idx_base_A,
                        hipsparseIndexBase_t idx_base_B,
                        hipsparseIndexBase_t idx_base_C,
                        hipsparseIndexBase_t idx_base_D)
{
    std::vector<int> nnz(n, -1);

    // Index base
    csr_row_ptr_C[0] = idx_base_C;

    // Loop over rows of A
    for(int i = 0; i < m; ++i)
    {
        // Initialize csr row pointer with previous row offset
        csr_row_ptr_C[i + 1] = csr_row_ptr_C[i];

        int row_begin_A = csr_row_ptr_A[i] - idx_base_A;
        int row_end_A   = csr_row_ptr_A[i + 1] - idx_base_A;

        // Loop over columns of A
        for(int j = row_begin_A; j < row_end_A; ++j)
        {
            // Current column of A
            int col_A = csr_col_ind_A[j] - idx_base_A;

            int row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
            int row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

            // Loop over columns of B in row col_A
            for(int k = row_begin_B; k < row_end_B; ++k)
            {
                // Current column of B
                int col_B = csr_col_ind_B[k] - idx_base_B;

                // Check if a new nnz is generated
                if(nnz[col_B] != i)
                {
                    nnz[col_B] = i;
                    ++csr_row_ptr_C[i + 1];
                }
            }
        }

        // Add nnz of D if beta != 0
        if(beta != (T)0)
        {
            int row_begin_D = csr_row_ptr_D[i] - idx_base_D;
            int row_end_D   = csr_row_ptr_D[i + 1] - idx_base_D;

            // Loop over columns of D
            for(int j = row_begin_D; j < row_end_D; ++j)
            {
                int col_D = csr_col_ind_D[j] - idx_base_D;

                // Check if a new nnz is generated
                if(nnz[col_D] != i)
                {
                    nnz[col_D] = i;
                    ++csr_row_ptr_C[i + 1];
                }
            }
        }
    }

    return csr_row_ptr_C[m] - idx_base_C;
}

template <typename T>
static void csrgemm2(int                  m,
                     int                  n,
                     int                  k,
                     T                    alpha,
                     const int*           csr_row_ptr_A,
                     const int*           csr_col_ind_A,
                     const T*             csr_val_A,
                     const int*           csr_row_ptr_B,
                     const int*           csr_col_ind_B,
                     const T*             csr_val_B,
                     T                    beta,
                     const int*           csr_row_ptr_D,
                     const int*           csr_col_ind_D,
                     const T*             csr_val_D,
                     const int*           csr_row_ptr_C,
                     int*                 csr_col_ind_C,
                     T*                   csr_val_C,
                     hipsparseIndexBase_t idx_base_A,
                     hipsparseIndexBase_t idx_base_B,
                     hipsparseIndexBase_t idx_base_C,
                     hipsparseIndexBase_t idx_base_D)
{
    std::vector<int> nnz(n, -1);

    // Loop over rows of A
    for(int i = 0; i < m; ++i)
    {
        int row_begin_A = csr_row_ptr_A[i] - idx_base_A;
        int row_end_A   = csr_row_ptr_A[i + 1] - idx_base_A;

        int row_begin_C = csr_row_ptr_C[i] - idx_base_C;
        int row_end_C   = row_begin_C;

        // Loop over columns of A
        for(int j = row_begin_A; j < row_end_A; ++j)
        {
            // Current column of A
            int col_A = csr_col_ind_A[j] - idx_base_A;
            // Current value of A
            T val_A = alpha * csr_val_A[j];

            int row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
            int row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

            // Loop over columns of B in row col_A
            for(int k = row_begin_B; k < row_end_B; ++k)
            {
                // Current column of B
                int col_B = csr_col_ind_B[k] - idx_base_B;
                // Current value of B
                T val_B = csr_val_B[k];

                // Check if a new nnz is generated or if the product is appended
                if(nnz[col_B] < row_begin_C)
                {
                    nnz[col_B]               = row_end_C;
                    csr_col_ind_C[row_end_C] = col_B + idx_base_C;
                    csr_val_C[row_end_C]     = val_A * val_B;
                    ++row_end_C;
                }
                else
                {
                    csr_val_C[nnz[col_B]] += val_A * val_B;
                }
            }
        }

        // Add nnz of D if beta != 0
        if(beta != (T)0)
        {
            int row_begin_D = csr_row_ptr_D[i] - idx_base_D;
            int row_end_D   = csr_row_ptr_D[i + 1] - idx_base_D;

            // Loop over columns of D
            for(int j = row_begin_D; j < row_end_D; ++j)
            {
                // Current column of D
                int col_D = csr_col_ind_D[j] - idx_base_D;
                // Current value of D
                T val_D = beta * csr_val_D[j];

                // Check if a new nnz is generated or if the value is added
                if(nnz[col_D] < row_begin_C)
                {
                    nnz[col_D] = row_end_C;

                    csr_col_ind_C[row_end_C] = col_D + idx_base_D;
                    csr_val_C[row_end_C]     = val_D;
                    ++row_end_C;
                }
                else
                {
                    csr_val_C[nnz[col_D]] += val_D;
                }
            }
        }
    }

    // Sort column indices within each row
    for(int i = 0; i < m; ++i)
    {
        int row_begin = csr_row_ptr_C[i] - idx_base_C;
        int row_end   = csr_row_ptr_C[i + 1] - idx_base_C;

        for(int j = row_begin; j < row_end; ++j)
        {
            for(int jj = row_begin; jj < row_end - 1; ++jj)
            {
                if(csr_col_ind_C[jj] > csr_col_ind_C[jj + 1])
                {
                    // swap elements
                    int ind = csr_col_ind_C[jj];
                    T   val = csr_val_C[jj];

                    csr_col_ind_C[jj] = csr_col_ind_C[jj + 1];
                    csr_val_C[jj]     = csr_val_C[jj + 1];

                    csr_col_ind_C[jj + 1] = ind;
                    csr_val_C[jj + 1]     = val;
                }
            }
        }
    }
}

template <typename T>
hipsparseStatus_t testing_csrgemm2(Arguments argus)
{
    int                  safe_size  = 100;
    int                  M          = argus.M;
    int                  N          = argus.N;
    int                  K          = argus.K;
    hipsparseIndexBase_t idx_base_A = argus.idx_base;
    hipsparseIndexBase_t idx_base_B = argus.idx_base2;
    hipsparseIndexBase_t idx_base_C = argus.idx_base3;
    std::string          binfile    = "";
    std::string          filename   = "";
    T                    h_alpha    = argus.alpha;
    hipsparseStatus_t    status;
    size_t               size;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    if(M == -99 && N == -99 && K == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        M = N = K = safe_size;
    }

    if(argus.timing == 1)
    {
        filename = argus.filename;
    }

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr_A(new descr_struct);
    hipsparseMatDescr_t           descr_A = test_descr_A->descr;

    std::unique_ptr<descr_struct> test_descr_B(new descr_struct);
    hipsparseMatDescr_t           descr_B = test_descr_B->descr;

    std::unique_ptr<descr_struct> test_descr_C(new descr_struct);
    hipsparseMatDescr_t           descr_C = test_descr_C->descr;

    std::unique_ptr<csrgemm2_struct> unique_ptr_csrgemm2(new csrgemm2_struct);
    csrgemm2Info_t                   info = unique_ptr_csrgemm2->info;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_A, idx_base_A));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_B, idx_base_B));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_C, idx_base_C));

    // Determine number of non-zero elements
    double scale = 0.02;
    if(M > 1000 || K > 1000)
    {
        scale = 2.0 / std::max(M, K);
    }
    int nnz_A = M * scale * K;

    scale = 0.02;
    if(K > 1000 || N > 1000)
    {
        scale = 2.0 / std::max(K, N);
    }
    int nnz_B = K * scale * N;

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || K <= 0 || nnz_A <= 0 || nnz_B <= 0)
    {
        auto dAptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dAcol_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dAval_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dBptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dBcol_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dBval_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dCptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dCcol_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dCval_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dbuffer_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

        int*  dAptr   = (int*)dAptr_managed.get();
        int*  dAcol   = (int*)dAcol_managed.get();
        T*    dAval   = (T*)dAval_managed.get();
        int*  dBptr   = (int*)dBptr_managed.get();
        int*  dBcol   = (int*)dBcol_managed.get();
        T*    dBval   = (T*)dBval_managed.get();
        int*  dCptr   = (int*)dCptr_managed.get();
        int*  dCcol   = (int*)dCcol_managed.get();
        T*    dCval   = (T*)dCval_managed.get();
        void* dbuffer = (void*)dbuffer_managed.get();

        if(!dAval || !dAptr || !dAcol || !dBval || !dBptr || !dBcol || !dCval || !dCptr || !dCcol
           || !dbuffer)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!dAptr || !dAcol || !dAval || "
                                            "!dBptr || !dBcol || !dBval || "
                                            "!dCptr || !dCcol || !dCval || "
                                            "!dbuffer");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        // Test hipsparseXcsrgemm2_bufferSizeExt
        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  K,
                                                  &h_alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  dAptr,
                                                  dAcol,
                                                  descr_B,
                                                  nnz_B,
                                                  dBptr,
                                                  dBcol,
                                                  (T*)nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  info,
                                                  &size);

        if(M < 0 || N < 0 || K < 0 || nnz_A < 0 || nnz_B < 0)
        {
            verify_hipsparse_status_invalid_size(
                status, "Error: M < 0 || N < 0 || K < 0 || nnz_A < 0 || nnz_B < 0");
        }
        else
        {
            verify_hipsparse_status_success(
                status, "M >= 0 && N >= 0 && K >= 0 && nnz_A >= 0 && nnz_B >= 0");
        }

        // Test hipsparseXcsrgemm2Nnz
        int nnz_C;
        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       K,
                                       descr_A,
                                       nnz_A,
                                       dAptr,
                                       dAcol,
                                       descr_B,
                                       nnz_B,
                                       dBptr,
                                       dBcol,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);

        if(M < 0 || N < 0 || K < 0 || nnz_A < 0 || nnz_B < 0)
        {
            verify_hipsparse_status_invalid_size(
                status, "Error: M < 0 || N < 0 || K < 0 || nnz_A < 0 || nnz_B < 0");
        }
        else
        {
            verify_hipsparse_status_success(
                status, "M >= 0 && N >= 0 && K >= 0 && nnz_A >= 0 && nnz_B >= 0");
        }

        // Test hipsparseXcsrgemm2
        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    K,
                                    &h_alpha,
                                    descr_A,
                                    nnz_A,
                                    dAval,
                                    dAptr,
                                    dAcol,
                                    descr_B,
                                    nnz_B,
                                    dBval,
                                    dBptr,
                                    dBcol,
                                    (T*)nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);

        if(M < 0 || N < 0 || K < 0 || nnz_A < 0 || nnz_B < 0)
        {
            verify_hipsparse_status_invalid_size(
                status, "Error: M < 0 || N < 0 || K < 0 || nnz_A < 0 || nnz_B < 0");
        }
        else
        {
            verify_hipsparse_status_success(
                status, "M >= 0 && N >= 0 && K >= 0 && nnz_A >= 0 && nnz_B >= 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Host structures
    std::vector<int> hcsr_row_ptr_A;
    std::vector<int> hcsr_col_ind_A;
    std::vector<T>   hcsr_val_A;

    // Initial Data on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(
               binfile.c_str(), M, K, nnz_A, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idx_base_A)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }
    else if(argus.laplacian)
    {
        M = K = gen_2d_laplacian(
            argus.laplacian, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idx_base_A);
        nnz_A = hcsr_row_ptr_A[M];
    }
    else
    {
        std::vector<int> hcoo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(filename.c_str(),
                               M,
                               K,
                               nnz_A,
                               hcoo_row_ind,
                               hcsr_col_ind_A,
                               hcsr_val_A,
                               idx_base_A)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            gen_matrix_coo(M, K, nnz_A, hcoo_row_ind, hcsr_col_ind_A, hcsr_val_A, idx_base_A);
        }

        // Convert COO to CSR
        hcsr_row_ptr_A.resize(M + 1, 0);
        for(int i = 0; i < nnz_A; ++i)
        {
            ++hcsr_row_ptr_A[hcoo_row_ind[i] + 1 - idx_base_A];
        }

        hcsr_row_ptr_A[0] = idx_base_A;
        for(int i = 0; i < M; ++i)
        {
            hcsr_row_ptr_A[i + 1] += hcsr_row_ptr_A[i];
        }

        // TODO samples B matrix instead of squaring
    }

    // B = A^T so that we can compute the square of A
    N     = M;
    nnz_B = nnz_A;
    std::vector<int> hcsr_row_ptr_B(K + 1, 0);
    std::vector<int> hcsr_col_ind_B(nnz_B);
    std::vector<T>   hcsr_val_B(nnz_B);

    // B = A^T
    transpose(M,
              K,
              nnz_A,
              hcsr_row_ptr_A.data(),
              hcsr_col_ind_A.data(),
              hcsr_val_A.data(),
              hcsr_row_ptr_B.data(),
              hcsr_col_ind_B.data(),
              hcsr_val_B.data(),
              idx_base_A,
              idx_base_B);

    // For simplicity, we generate a COO matrix for D
    std::vector<int> hcsr_row_ptr_D;
    std::vector<int> hcsr_col_ind_D;
    std::vector<T>   hcsr_val_D;

    // Allocate memory on device
    auto dAptr_managed  = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};
    auto dAcol_managed  = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_A), device_free};
    auto dAval_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dBptr_managed  = hipsparse_unique_ptr{device_malloc(sizeof(int) * (K + 1)), device_free};
    auto dBcol_managed  = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_B), device_free};
    auto dBval_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dCptr_managed  = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};
    auto dalpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int* dAptr  = (int*)dAptr_managed.get();
    int* dAcol  = (int*)dAcol_managed.get();
    T*   dAval  = (T*)dAval_managed.get();
    int* dBptr  = (int*)dBptr_managed.get();
    int* dBcol  = (int*)dBcol_managed.get();
    T*   dBval  = (T*)dBval_managed.get();
    int* dCptr  = (int*)dCptr_managed.get();
    T*   dalpha = (T*)dalpha_managed.get();

    if(!dAval || !dAptr || !dAcol || !dBval || !dBptr || !dBcol || !dCptr || !dalpha)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dAval || !dAptr || !dAcol || "
                                        "!dBval || !dBptr || !dBcol || "
                                        "!dCptr || !dalpha");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dAptr, hcsr_row_ptr_A.data(), sizeof(int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dAcol, hcsr_col_ind_A.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dAval, hcsr_val_A.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dBptr, hcsr_row_ptr_B.data(), sizeof(int) * (K + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dBcol, hcsr_col_ind_B.data(), sizeof(int) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dBval, hcsr_val_B.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dalpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // Obtain csrgemm2 buffer size
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                           M,
                                                           N,
                                                           K,
                                                           &h_alpha,
                                                           descr_A,
                                                           nnz_A,
                                                           dAptr,
                                                           dAcol,
                                                           descr_B,
                                                           nnz_B,
                                                           dBptr,
                                                           dBcol,
                                                           (T*)nullptr,
                                                           nullptr,
                                                           0,
                                                           nullptr,
                                                           nullptr,
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

    // csrgemm2 nnz

    // hipsparse pointer mode host
    int hnnz_C_1;

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm2Nnz(handle,
                                                M,
                                                N,
                                                K,
                                                descr_A,
                                                nnz_A,
                                                dAptr,
                                                dAcol,
                                                descr_B,
                                                nnz_B,
                                                dBptr,
                                                dBcol,
                                                nullptr,
                                                0,
                                                nullptr,
                                                nullptr,
                                                descr_C,
                                                dCptr,
                                                &hnnz_C_1,
                                                info,
                                                dbuffer));

    // Allocate result matrix
    auto dCcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * hnnz_C_1), device_free};
    auto dCval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * hnnz_C_1), device_free};

    int* dCcol = (int*)dCcol_managed.get();
    T*   dCval = (T*)dCval_managed.get();

    if(!dCval || !dCcol)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED, "!dCval || !dCcol");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    if(argus.unit_check)
    {
        // hipsparse pointer mode device
        auto dnnz_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
        int* dnnz_C         = (int*)dnnz_C_managed.get();

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm2Nnz(handle,
                                                    M,
                                                    N,
                                                    K,
                                                    descr_A,
                                                    nnz_A,
                                                    dAptr,
                                                    dAcol,
                                                    descr_B,
                                                    nnz_B,
                                                    dBptr,
                                                    dBcol,
                                                    nullptr,
                                                    0,
                                                    nullptr,
                                                    nullptr,
                                                    descr_C,
                                                    dCptr,
                                                    dnnz_C,
                                                    info,
                                                    dbuffer));

        // Compute csrgemm host solution
        std::vector<int> hcsr_row_ptr_C_gold(M + 1);

        double cpu_time_used = get_time_us();

        int nnz_C_gold = csrgemm2_nnz(M,
                                      N,
                                      K,
                                      hcsr_row_ptr_A.data(),
                                      hcsr_col_ind_A.data(),
                                      hcsr_row_ptr_B.data(),
                                      hcsr_col_ind_B.data(),
                                      (T)0.0,
                                      hcsr_row_ptr_D.data(),
                                      hcsr_col_ind_D.data(),
                                      hcsr_row_ptr_C_gold.data(),
                                      idx_base_A,
                                      idx_base_B,
                                      idx_base_C,
                                      HIPSPARSE_INDEX_BASE_ZERO);

        std::vector<int> hcsr_col_ind_C_gold(nnz_C_gold);
        std::vector<T>   hcsr_val_C_gold(nnz_C_gold);

        csrgemm2(M,
                 N,
                 K,
                 h_alpha,
                 hcsr_row_ptr_A.data(),
                 hcsr_col_ind_A.data(),
                 hcsr_val_A.data(),
                 hcsr_row_ptr_B.data(),
                 hcsr_col_ind_B.data(),
                 hcsr_val_B.data(),
                 (T)0.0,
                 hcsr_row_ptr_D.data(),
                 hcsr_col_ind_D.data(),
                 hcsr_val_D.data(),
                 hcsr_row_ptr_C_gold.data(),
                 hcsr_col_ind_C_gold.data(),
                 hcsr_val_C_gold.data(),
                 idx_base_A,
                 idx_base_B,
                 idx_base_C,
                 HIPSPARSE_INDEX_BASE_ZERO);

        cpu_time_used = get_time_us() - cpu_time_used;

        // Copy output from device to CPU
        int hnnz_C_2;
        CHECK_HIP_ERROR(hipMemcpy(&hnnz_C_2, dnnz_C, sizeof(int), hipMemcpyDeviceToHost));

        // Check nnz of C
        unit_check_general(1, 1, 1, &nnz_C_gold, &hnnz_C_1);
        unit_check_general(1, 1, 1, &nnz_C_gold, &hnnz_C_2);

        // Compute csrgemm2
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm2(handle,
                                                 M,
                                                 N,
                                                 K,
                                                 &h_alpha,
                                                 descr_A,
                                                 nnz_A,
                                                 dAval,
                                                 dAptr,
                                                 dAcol,
                                                 descr_B,
                                                 nnz_B,
                                                 dBval,
                                                 dBptr,
                                                 dBcol,
                                                 (T*)nullptr,
                                                 nullptr,
                                                 0,
                                                 (T*)nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descr_C,
                                                 dCval,
                                                 dCptr,
                                                 dCcol,
                                                 info,
                                                 dbuffer));

        // Copy output from device to CPU
        std::vector<int> hcsr_row_ptr_C(M + 1);
        std::vector<int> hcsr_col_ind_C(nnz_C_gold);
        std::vector<T>   hcsr_val_C(nnz_C_gold);

        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_row_ptr_C.data(), dCptr, sizeof(int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind_C.data(), dCcol, sizeof(int) * nnz_C_gold, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C.data(), dCval, sizeof(T) * nnz_C_gold, hipMemcpyDeviceToHost));

        // Check structure and entries of C
        unit_check_general(1, M + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C.data());
        //        unit_check_general(1, nnz_C_gold, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C.data());
        //        unit_check_near(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C.data());
        //        unit_check_general(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C.data());
    }

    if(argus.timing)
    {
        // TODO
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSRGEMM2_HPP
