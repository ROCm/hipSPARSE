/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef TESTING_GEMVI_HPP
#define TESTING_GEMVI_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <typeinfo>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_gemvi_bad_arg(void)
{
    int m   = 100;
    int n   = 100;
    int nnz = 100;
    int lda = 100;

    hipsparseOperation_t opType  = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idxBase = HIPSPARSE_INDEX_BASE_ZERO;

    float alpha = 0.6;
    float beta  = 0.1;

    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto A_managed    = hipsparse_unique_ptr{device_malloc(sizeof(float) * m * n), device_free};
    auto x_managed    = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto xInd_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto y_managed    = hipsparse_unique_ptr{device_malloc(sizeof(float) * m), device_free};

    float* A    = (float*)A_managed.get();
    float* x    = (float*)x_managed.get();
    int*   xInd = (int*)xInd_managed.get();
    float* y    = (float*)y_managed.get();

    if(!A || !xInd || !x || !y)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // cusparse has error checks for this function at all
#if(!defined(CUDART_VERSION))
    // gemvi bufferSize - buffer size is currently not doing anything
    //    int bufferSize;
    //    verify_hipsparse_status_invalid_handle(
    //        hipsparseSpVV_bufferSize(nullptr, opType, x, y, &result, dataType, &bufferSize));
    //    verify_hipsparse_status_invalid_pointer(
    //        hipsparseSpVV_bufferSize(handle, opType, nullptr, y, &result, dataType, &bufferSize),
    //        "Error: x is nullptr");
    //    verify_hipsparse_status_invalid_pointer(
    //        hipsparseSpVV_bufferSize(handle, opType, x, nullptr, &result, dataType, &bufferSize),
    //        "Error: y is nullptr");
    //    verify_hipsparse_status_invalid_pointer(
    //        hipsparseSpVV_bufferSize(handle, opType, x, y, nullptr, dataType, &bufferSize),
    //        "Error: result is nullptr");
    //    verify_hipsparse_status_invalid_pointer(
    //        hipsparseSpVV_bufferSize(handle, opType, x, y, &result, dataType, nullptr),
    //        "Error: bufferSize is nullptr");

    // gemvi
    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, 100));

    verify_hipsparse_status_invalid_handle(hipsparseSgemvi(
        nullptr, opType, m, n, &alpha, A, lda, nnz, x, xInd, &beta, y, idxBase, buffer));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSgemvi(
            handle, opType, m, n, nullptr, A, lda, nnz, x, xInd, &beta, y, idxBase, buffer),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, nullptr, lda, nnz, x, xInd, &beta, y, idxBase, buffer),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, lda, nnz, nullptr, xInd, &beta, y, idxBase, buffer),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, lda, nnz, x, nullptr, &beta, y, idxBase, buffer),
        "Error: xInd is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, lda, nnz, x, xInd, nullptr, y, idxBase, buffer),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, lda, nnz, x, xInd, &beta, nullptr, idxBase, buffer),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, lda, nnz, x, xInd, &beta, y, idxBase, nullptr),
        "Error: buffer is nullptr");

    verify_hipsparse_status_invalid_size(
        hipsparseSgemvi(
            handle, opType, -1, n, &alpha, A, lda, nnz, x, xInd, &beta, y, idxBase, buffer),
        "Error: m is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseSgemvi(
            handle, opType, m, -1, &alpha, A, lda, nnz, x, xInd, &beta, y, idxBase, buffer),
        "Error: n is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, -1, nnz, x, xInd, &beta, y, idxBase, buffer),
        "Error: lda is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, lda, -1, x, xInd, &beta, y, idxBase, buffer),
        "Error: nnz is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, lda, n + 1, x, xInd, &beta, y, idxBase, buffer),
        "Error: nnz is invalid");

    CHECK_HIP_ERROR(hipFree(buffer));
#endif
}

template <typename T>
hipsparseStatus_t testing_gemvi(void)
{
    int m   = 1291;
    int n   = 724;
    int nnz = 237;

    hipsparseOperation_t opType  = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idxBase = HIPSPARSE_INDEX_BASE_ZERO;

    T alpha = make_DataType<T>(0.6);
    T beta  = make_DataType<T>(3.2);

    int lda = (opType == HIPSPARSE_OPERATION_NON_TRANSPOSE ? m : n);

    hipsparseStatus_t status;

    // hipSPARSE handle
    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    // Host structures
    std::vector<T>   hA(m * n);
    std::vector<T>   hx_val(nnz);
    std::vector<int> hx_ind(nnz);
    std::vector<T>   hy(m);
    std::vector<T>   hy_gold(m);

    // Initial Data on CPU
    srand(12345ULL);
    hipsparseInitIndex(hx_ind.data(), nnz, 1, n);
    hipsparseInit<T>(hx_val, 1, nnz);
    hipsparseInit<T>(hy, 1, m);
    hy_gold = hy;

    for(int i = 0; i < m * n; ++i)
    {
        hA[i] = random_generator<T>();
    }

    // Allocate memory on device
    auto dx_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dx_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dA_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * n), device_free};

    int* dx_ind = (int*)dx_ind_managed.get();
    T*   dx_val = (T*)dx_val_managed.get();
    T*   dy     = (T*)dy_managed.get();
    T*   dA     = (T*)dA_managed.get();

    if(!dx_ind || !dx_val || !dy || !dA)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dx_ind || !dx_val || !dy || !dA");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * m * n, hipMemcpyHostToDevice));

    // gemvi bufferSize
    int   bufferSize;
    void* externalBuffer;

    CHECK_HIPSPARSE_ERROR(hipsparseXgemvi_bufferSize<T>(handle, opType, m, n, nnz, &bufferSize));
    CHECK_HIP_ERROR(hipMalloc(&externalBuffer, bufferSize));

    // gemvi
    CHECK_HIPSPARSE_ERROR(hipsparseXgemvi(handle,
                                          opType,
                                          m,
                                          n,
                                          &alpha,
                                          dA,
                                          lda,
                                          nnz,
                                          dx_val,
                                          dx_ind,
                                          &beta,
                                          dy,
                                          idxBase,
                                          externalBuffer));
    CHECK_HIP_ERROR(hipFree(externalBuffer));

    // CPU
    for(int i = 0; i < m; ++i)
    {
        T sum = make_DataType<T>(0);

        for(int j = 0; j < nnz; ++j)
        {
            sum = testing_fma(hx_val[j], hA[hx_ind[j] * lda + i], sum);
        }

        hy_gold[i] = testing_fma(alpha, sum, beta * hy_gold[i]);
    }

    // Verify results against host
    CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * m, hipMemcpyDeviceToHost));

    unit_check_near(m, 1, 1, hy_gold.data(), hy.data());

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_GEMVI_HPP
