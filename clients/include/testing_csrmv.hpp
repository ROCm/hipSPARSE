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
#ifndef TESTING_CSRMV_HPP
#define TESTING_CSRMV_HPP

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
void testing_csrmv_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                  n         = 100;
    int                  m         = 100;
    int                  nnz       = 100;
    int                  safe_size = 100;
    T                    alpha     = 0.6;
    T                    beta      = 0.2;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseStatus_t    status;

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

    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXcsrmv(
            handle, transA, m, n, nnz, &alpha, descr, dval, dptr_null, dcol, dx, &beta, dy);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXcsrmv(
            handle, transA, m, n, nnz, &alpha, descr, dval, dptr, dcol_null, dx, &beta, dy);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXcsrmv(
            handle, transA, m, n, nnz, &alpha, descr, dval_null, dptr, dcol, dx, &beta, dy);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == dx)
    {
        T* dx_null = nullptr;

        status = hipsparseXcsrmv(
            handle, transA, m, n, nnz, &alpha, descr, dval, dptr, dcol, dx_null, &beta, dy);
        verify_hipsparse_status_invalid_pointer(status, "Error: dx is nullptr");
    }
    // testing for(nullptr == dy)
    {
        T* dy_null = nullptr;

        status = hipsparseXcsrmv(
            handle, transA, m, n, nnz, &alpha, descr, dval, dptr, dcol, dx, &beta, dy_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: dy is nullptr");
    }
    // testing for(nullptr == d_alpha)
    {
        T* d_alpha_null = nullptr;

        status = hipsparseXcsrmv(
            handle, transA, m, n, nnz, d_alpha_null, descr, dval, dptr, dcol, dx, &beta, dy);
        verify_hipsparse_status_invalid_pointer(status, "Error: alpha is nullptr");
    }
    // testing for(nullptr == d_beta)
    {
        T* d_beta_null = nullptr;

        status = hipsparseXcsrmv(
            handle, transA, m, n, nnz, &alpha, descr, dval, dptr, dcol, dx, d_beta_null, dy);
        verify_hipsparse_status_invalid_pointer(status, "Error: beta is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXcsrmv(
            handle, transA, m, n, nnz, &alpha, descr_null, dval, dptr, dcol, dx, &beta, dy);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrmv(
            handle_null, transA, m, n, nnz, &alpha, descr, dval, dptr, dcol, dx, &beta, dy);
        verify_hipsparse_status_invalid_handle(status);
    }
#endif
}

template <typename T>
hipsparseStatus_t testing_csrmv(Arguments argus)
{
    int                  nrow     = argus.M;
    int                  ncol     = argus.N;
    T                    h_alpha  = make_DataType<T>(argus.alpha);
    T                    h_beta   = make_DataType<T>(argus.beta);
    hipsparseOperation_t transA   = argus.transA;
    hipsparseIndexBase_t idx_base = argus.idx_base;
    std::string          filename = argus.filename;

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr(new descr_struct);
    hipsparseMatDescr_t           descr = test_descr->descr;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_csr_matrix(
           filename, nrow, ncol, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // std::cout << "nrow: " << nrow << " ncol: " << ncol << " nnz: " << nnz << std::endl;
    // std::cout << "A" << std::endl;
    // for(int i = 0; i < nrow; i++)
    // {
    //     int start = hcsr_row_ptr[i] - idx_base;
    //     int end = hcsr_row_ptr[i + 1] - idx_base;

    //     std::vector<double> temp(ncol, 0.0);
    //     for(int j = start; j < end; j++)
    //     {
    //         temp[hcsr_col_ind[j] - idx_base] = 1.0;//(double)hcsr_val[j];
    //     }

    //     for(size_t j = 0; j < temp.size(); j++)
    //     {
    //         std::cout << temp[j] << " ";
    //     }
    //     std::cout << "" << std::endl;
    // }

    int m = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? nrow : ncol;
    int n = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? ncol : nrow;

    std::vector<T> hx(n);
    std::vector<T> hy_1(m);
    std::vector<T> hy_2(m);
    std::vector<T> hy_gold(m);

    hipsparseInit<T>(hx, 1, n);
    hipsparseInit<T>(hy_1, 1, m);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (nrow + 1)), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * n), device_free};
    auto dy_1_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int* dptr    = (int*)dptr_managed.get();
    int* dcol    = (int*)dcol_managed.get();
    T*   dval    = (T*)dval_managed.get();
    T*   dx      = (T*)dx_managed.get();
    T*   dy_1    = (T*)dy_1_managed.get();
    T*   dy_2    = (T*)dy_2_managed.get();
    T*   d_alpha = (T*)d_alpha_managed.get();
    T*   d_beta  = (T*)d_beta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(int) * (nrow + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * m, hipMemcpyHostToDevice));

        // ROCSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrmv(
            handle, transA, nrow, ncol, nnz, &h_alpha, descr, dval, dptr, dcol, dx, &h_beta, dy_1));

        // ROCSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrmv(
            handle, transA, nrow, ncol, nnz, d_alpha, descr, dval, dptr, dcol, dx, d_beta, dy_2));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * m, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * m, hipMemcpyDeviceToHost));

        host_csrmv(transA,
                   nrow,
                   ncol,
                   nnz,
                   h_alpha,
                   hcsr_row_ptr.data(),
                   hcsr_col_ind.data(),
                   hcsr_val.data(),
                   hx.data(),
                   h_beta,
                   hy_gold.data(),
                   idx_base);

        unit_check_near(1, m, 1, hy_gold.data(), hy_1.data());
        unit_check_near(1, m, 1, hy_gold.data(), hy_2.data());
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSRMV_HPP
