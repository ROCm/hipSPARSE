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
#ifndef TESTING_CSRGEMM_HPP
#define TESTING_CSRGEMM_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_csrgemm_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVCC__
    // do not test for bad args
    return;
#endif
    int                  M         = 100;
    int                  N         = 100;
    int                  K         = 100;
    int                  nnz_A     = 100;
    int                  nnz_B     = 100;
    hipsparseOperation_t trans_A   = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t trans_B   = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    int                  safe_size = 100;
    hipsparseStatus_t    status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr_A(new descr_struct);
    hipsparseMatDescr_t           descr_A = unique_ptr_descr_A->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_B(new descr_struct);
    hipsparseMatDescr_t           descr_B = unique_ptr_descr_B->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_C(new descr_struct);
    hipsparseMatDescr_t           descr_C = unique_ptr_descr_C->descr;

    auto dAptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dAcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dAval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dBptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dBcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dBval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dCptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dCcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dCval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* dAptr = (int*)dAptr_managed.get();
    int* dAcol = (int*)dAcol_managed.get();
    T*   dAval = (T*)dAval_managed.get();
    int* dBptr = (int*)dBptr_managed.get();
    int* dBcol = (int*)dBcol_managed.get();
    T*   dBval = (T*)dBval_managed.get();
    int* dCptr = (int*)dCptr_managed.get();
    int* dCcol = (int*)dCcol_managed.get();
    T*   dCval = (T*)dCval_managed.get();

    if(!dAval || !dAptr || !dAcol || !dBval || !dBptr || !dBcol || !dCval || !dCptr || !dCcol)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing hipsparseXcsrgemmNnz
    int nnz_C;

    // testing for(nullptr == dAptr)
    {
        int* dAptr_null = nullptr;

        status = hipsparseXcsrgemmNnz(handle,
                                      trans_A,
                                      trans_B,
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
                                      descr_C,
                                      dCptr,
                                      &nnz_C);
        verify_hipsparse_status_invalid_pointer(status, "Error: dAptr is nullptr");
    }
    // testing for(nullptr == dAcol)
    {
        int* dAcol_null = nullptr;

        status = hipsparseXcsrgemmNnz(handle,
                                      trans_A,
                                      trans_B,
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
                                      descr_C,
                                      dCptr,
                                      &nnz_C);
        verify_hipsparse_status_invalid_pointer(status, "Error: dAcol is nullptr");
    }
    // testing for(nullptr == dBptr)
    {
        int* dBptr_null = nullptr;

        status = hipsparseXcsrgemmNnz(handle,
                                      trans_A,
                                      trans_B,
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
                                      descr_C,
                                      dCptr,
                                      &nnz_C);
        verify_hipsparse_status_invalid_pointer(status, "Error: dBptr is nullptr");
    }
    // testing for(nullptr == dBcol)
    {
        int* dBcol_null = nullptr;

        status = hipsparseXcsrgemmNnz(handle,
                                      trans_A,
                                      trans_B,
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
                                      descr_C,
                                      dCptr,
                                      &nnz_C);
        verify_hipsparse_status_invalid_pointer(status, "Error: dBcol is nullptr");
    }
    // testing for(nullptr == dCptr)
    {
        int* dCptr_null = nullptr;

        status = hipsparseXcsrgemmNnz(handle,
                                      trans_A,
                                      trans_B,
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
                                      descr_C,
                                      dCptr_null,
                                      &nnz_C);
        verify_hipsparse_status_invalid_pointer(status, "Error: dCptr is nullptr");
    }
    // testing for(nullptr == nnz_C)
    {
        int* nnz_C_null = nullptr;

        status = hipsparseXcsrgemmNnz(handle,
                                      trans_A,
                                      trans_B,
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
                                      descr_C,
                                      dCptr,
                                      nnz_C_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: nnz_C is nullptr");
    }
    // testing for(nullptr == descr_A)
    {
        hipsparseMatDescr_t descr_A_null = nullptr;

        status = hipsparseXcsrgemmNnz(handle,
                                      trans_A,
                                      trans_B,
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
                                      descr_C,
                                      dCptr,
                                      &nnz_C);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");
    }
    // testing for(nullptr == descr_B)
    {
        hipsparseMatDescr_t descr_B_null = nullptr;

        status = hipsparseXcsrgemmNnz(handle,
                                      trans_A,
                                      trans_B,
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
                                      descr_C,
                                      dCptr,
                                      &nnz_C);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_B is nullptr");
    }
    // testing for(nullptr == descr_C)
    {
        hipsparseMatDescr_t descr_C_null = nullptr;

        status = hipsparseXcsrgemmNnz(handle,
                                      trans_A,
                                      trans_B,
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
                                      descr_C_null,
                                      dCptr,
                                      &nnz_C);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrgemmNnz(handle_null,
                                      trans_A,
                                      trans_B,
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
                                      descr_C,
                                      dCptr,
                                      &nnz_C);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing hipsparseXcsrgemm

    // testing for(nullptr == dAval)
    {
        T* dAval_null = nullptr;

        status = hipsparseXcsrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol);
        verify_hipsparse_status_invalid_pointer(status, "Error: dAval is nullptr");
    }
    // testing for(nullptr == dAptr)
    {
        int* dAptr_null = nullptr;

        status = hipsparseXcsrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol);
        verify_hipsparse_status_invalid_pointer(status, "Error: dAptr is nullptr");
    }
    // testing for(nullptr == dAcol)
    {
        int* dAcol_null = nullptr;

        status = hipsparseXcsrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol);
        verify_hipsparse_status_invalid_pointer(status, "Error: dAcol is nullptr");
    }
    // testing for(nullptr == dBval)
    {
        T* dBval_null = nullptr;

        status = hipsparseXcsrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol);
        verify_hipsparse_status_invalid_pointer(status, "Error: dBval is nullptr");
    }
    // testing for(nullptr == dBptr)
    {
        int* dBptr_null = nullptr;

        status = hipsparseXcsrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol);
        verify_hipsparse_status_invalid_pointer(status, "Error: dBptr is nullptr");
    }
    // testing for(nullptr == dBcol)
    {
        int* dBcol_null = nullptr;

        status = hipsparseXcsrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol);
        verify_hipsparse_status_invalid_pointer(status, "Error: dBcol is nullptr");
    }
    // testing for(nullptr == dCval)
    {
        T* dCval_null = nullptr;

        status = hipsparseXcsrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval_null,
                                   dCptr,
                                   dCcol);
        verify_hipsparse_status_invalid_pointer(status, "Error: dCval is nullptr");
    }
    // testing for(nullptr == dCptr)
    {
        int* dCptr_null = nullptr;

        status = hipsparseXcsrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr_null,
                                   dCcol);
        verify_hipsparse_status_invalid_pointer(status, "Error: dCptr is nullptr");
    }
    // testing for(nullptr == dCcol)
    {
        int* dCcol_null = nullptr;

        status = hipsparseXcsrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: dCcol is nullptr");
    }
    // testing for(nullptr == descr_A)
    {
        hipsparseMatDescr_t descr_A_null = nullptr;

        status = hipsparseXcsrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");
    }
    // testing for(nullptr == descr_B)
    {
        hipsparseMatDescr_t descr_B_null = nullptr;

        status = hipsparseXcsrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_B is nullptr");
    }
    // testing for(nullptr == descr_C)
    {
        hipsparseMatDescr_t descr_C_null = nullptr;

        status = hipsparseXcsrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C_null,
                                   dCval,
                                   dCptr,
                                   dCcol);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrgemm(handle_null,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol);
        verify_hipsparse_status_invalid_handle(status);
    }
}

static int csrgemm_nnz(int                  m,
                       int                  n,
                       int                  k,
                       const int*           csr_row_ptr_A,
                       const int*           csr_col_ind_A,
                       const int*           csr_row_ptr_B,
                       const int*           csr_col_ind_B,
                       int*                 csr_row_ptr_C,
                       hipsparseIndexBase_t idx_base_A,
                       hipsparseIndexBase_t idx_base_B,
                       hipsparseIndexBase_t idx_base_C)
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
    }

    return csr_row_ptr_C[m] - idx_base_C;
}

template <typename T>
static void csrgemm(int                  m,
                    int                  n,
                    int                  k,
                    const int*           csr_row_ptr_A,
                    const int*           csr_col_ind_A,
                    const T*             csr_val_A,
                    const int*           csr_row_ptr_B,
                    const int*           csr_col_ind_B,
                    const T*             csr_val_B,
                    const int*           csr_row_ptr_C,
                    int*                 csr_col_ind_C,
                    T*                   csr_val_C,
                    hipsparseIndexBase_t idx_base_A,
                    hipsparseIndexBase_t idx_base_B,
                    hipsparseIndexBase_t idx_base_C)
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
            T val_A = csr_val_A[j];

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
hipsparseStatus_t testing_csrgemm(Arguments argus)
{
    int                  safe_size  = 100;
    int                  M          = argus.M;
    int                  N          = argus.N;
    int                  K          = argus.K;
    hipsparseOperation_t trans_A    = argus.transA;
    hipsparseOperation_t trans_B    = argus.transB;
    hipsparseIndexBase_t idx_base_A = argus.idx_base;
    hipsparseIndexBase_t idx_base_B = argus.idx_base2;
    hipsparseIndexBase_t idx_base_C = argus.idx_base3;
    std::string          binfile    = "";
    std::string          filename   = "";
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

        int* dAptr = (int*)dAptr_managed.get();
        int* dAcol = (int*)dAcol_managed.get();
        T*   dAval = (T*)dAval_managed.get();
        int* dBptr = (int*)dBptr_managed.get();
        int* dBcol = (int*)dBcol_managed.get();
        T*   dBval = (T*)dBval_managed.get();
        int* dCptr = (int*)dCptr_managed.get();
        int* dCcol = (int*)dCcol_managed.get();
        T*   dCval = (T*)dCval_managed.get();

        if(!dAval || !dAptr || !dAcol || !dBval || !dBptr || !dBcol || !dCval || !dCptr || !dCcol)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!dAptr || !dAcol || !dAval || "
                                            "!dBptr || !dBcol || !dBval || "
                                            "!dCptr || !dCcol || !dCval");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        // Test hipsparseXcsrgemmNnz
        int nnz_C;
        status = hipsparseXcsrgemmNnz(handle,
                                      trans_A,
                                      trans_B,
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
                                      descr_C,
                                      dCptr,
                                      &nnz_C);

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

        // Test hipsparseXcsrgemm
        status = hipsparseXcsrgemm(handle,
                                   trans_A,
                                   trans_B,
                                   M,
                                   N,
                                   K,
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
                                   descr_C,
                                   dCval,
                                   dCptr,
                                   dCcol);

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

    // Allocate memory on device
    auto dAptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};
    auto dAcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_A), device_free};
    auto dAval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dBptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (K + 1)), device_free};
    auto dBcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_B), device_free};
    auto dBval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dCptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};

    int* dAptr = (int*)dAptr_managed.get();
    int* dAcol = (int*)dAcol_managed.get();
    T*   dAval = (T*)dAval_managed.get();
    int* dBptr = (int*)dBptr_managed.get();
    int* dBcol = (int*)dBcol_managed.get();
    T*   dBval = (T*)dBval_managed.get();
    int* dCptr = (int*)dCptr_managed.get();

    if(!dAval || !dAptr || !dAcol || !dBval || !dBptr || !dBcol || !dCptr)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dAval || !dAptr || !dAcol || "
                                        "!dBval || !dBptr || !dBcol || !dCptr");
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

    // csrgemm nnz

    // hipsparse pointer mode host
    int hnnz_C_1;

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemmNnz(handle,
                                               trans_A,
                                               trans_B,
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
                                               descr_C,
                                               dCptr,
                                               &hnnz_C_1));

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
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemmNnz(handle,
                                                   trans_A,
                                                   trans_B,
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
                                                   descr_C,
                                                   dCptr,
                                                   dnnz_C))

        // Compute csrgemm host solution
        std::vector<int> hcsr_row_ptr_C_gold(M + 1);

        double cpu_time_used = get_time_us();

        int nnz_C_gold = csrgemm_nnz(M,
                                     N,
                                     K,
                                     hcsr_row_ptr_A.data(),
                                     hcsr_col_ind_A.data(),
                                     hcsr_row_ptr_B.data(),
                                     hcsr_col_ind_B.data(),
                                     hcsr_row_ptr_C_gold.data(),
                                     idx_base_A,
                                     idx_base_B,
                                     idx_base_C);

        std::vector<int> hcsr_col_ind_C_gold(nnz_C_gold);
        std::vector<T>   hcsr_val_C_gold(nnz_C_gold);

        csrgemm(M,
                N,
                K,
                hcsr_row_ptr_A.data(),
                hcsr_col_ind_A.data(),
                hcsr_val_A.data(),
                hcsr_row_ptr_B.data(),
                hcsr_col_ind_B.data(),
                hcsr_val_B.data(),
                hcsr_row_ptr_C_gold.data(),
                hcsr_col_ind_C_gold.data(),
                hcsr_val_C_gold.data(),
                idx_base_A,
                idx_base_B,
                idx_base_C);

        cpu_time_used = get_time_us() - cpu_time_used;

        // Copy output from device to CPU
        int hnnz_C_2;
        CHECK_HIP_ERROR(hipMemcpy(&hnnz_C_2, dnnz_C, sizeof(int), hipMemcpyDeviceToHost));

        // Check nnz of C
        unit_check_general(1, 1, 1, &nnz_C_gold, &hnnz_C_1);
        unit_check_general(1, 1, 1, &nnz_C_gold, &hnnz_C_2);

        // Compute csrgemm
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm(handle,
                                                trans_A,
                                                trans_B,
                                                M,
                                                N,
                                                K,
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
                                                descr_C,
                                                dCval,
                                                dCptr,
                                                dCcol));

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
    }

    if(argus.timing)
    {
        // TODO
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSRGEMM_HPP
