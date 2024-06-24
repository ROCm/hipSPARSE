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
#ifndef TESTING_AXPBY_HPP
#define TESTING_AXPBY_HPP

#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include "hipsparse_arguments.hpp"

#include <hipsparse.h>
#include <typeinfo>

using namespace hipsparse_test;

void testing_axpby_bad_arg(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)

    int64_t size = 100;
    int64_t nnz  = 100;

    float alpha = 3.7f;
    float beta  = 1.2f;

    hipsparseIndexType_t idxType  = HIPSPARSE_INDEX_32I;
    hipsparseIndexBase_t idxBase  = HIPSPARSE_INDEX_BASE_ZERO;
    hipDataType          dataType = HIP_R_32F;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dx_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dx_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(float) * size), device_free};

    float* dx_val = (float*)dx_val_managed.get();
    int*   dx_ind = (int*)dx_ind_managed.get();
    float* dy     = (float*)dy_managed.get();

    if(!dx_ind || !dx_val || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Structures
    hipsparseSpVecDescr_t x;
    hipsparseDnVecDescr_t y;

    verify_hipsparse_status_success(
        hipsparseCreateSpVec(&x, size, nnz, dx_ind, dx_val, idxType, idxBase, dataType), "Success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&y, size, dy, dataType), "Success");

    // Axpby
    verify_hipsparse_status_invalid_handle(hipsparseAxpby(nullptr, &alpha, x, &beta, y));
    verify_hipsparse_status_invalid_pointer(hipsparseAxpby(handle, nullptr, x, &beta, y),
                                            "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseAxpby(handle, &alpha, nullptr, &beta, y),
                                            "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseAxpby(handle, &alpha, x, nullptr, y),
                                            "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseAxpby(handle, &alpha, x, &beta, nullptr),
                                            "Error: y is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpVec(x), "Success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(y), "Success");
#endif
}

template <typename I, typename T>
hipsparseStatus_t testing_axpby(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
    int64_t size = 15332;
    int64_t nnz  = 500;

    T alpha = make_DataType<T>(1.5);
    T beta  = make_DataType<T>(0.5);

    hipsparseIndexBase_t idxBase = HIPSPARSE_INDEX_BASE_ZERO;

    // Index and data type
    hipsparseIndexType_t idxType  = getIndexType<I>();
    hipDataType          dataType = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    // Host structures
    std::vector<I> hx_ind(nnz);
    std::vector<T> hx_val(nnz);
    std::vector<T> hy(size);
    std::vector<T> hy_gold(size);

    // Initial Data on CPU
    srand(12345ULL);
    hipsparseInitIndex(hx_ind.data(), nnz, 1, size);
    hipsparseInit<T>(hx_val, 1, nnz);
    hipsparseInit<T>(hy, 1, size);

    hy_gold = hy;

    // Allocate memory on device
    auto dx_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dx_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * size), device_free};

    I* dx_ind = (I*)dx_ind_managed.get();
    T* dx_val = (T*)dx_val_managed.get();
    T* dy     = (T*)dy_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * size, hipMemcpyHostToDevice));

    // Create structures
    hipsparseSpVecDescr_t x;
    hipsparseDnVecDescr_t y;

    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateSpVec(&x, size, nnz, dx_ind, dx_val, idxType, idxBase, dataType));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y, size, dy, dataType));

    // Axpby
    CHECK_HIPSPARSE_ERROR(hipsparseAxpby(handle, &alpha, x, &beta, y));

    // Copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * size, hipMemcpyDeviceToHost));

    // CPU
    for(int64_t i = 0; i < size; ++i)
    {
        hy_gold[i] = testing_mult(beta, hy_gold[i]);
    }

    for(int64_t i = 0; i < nnz; ++i)
    {
        hy_gold[hx_ind[i] - idxBase] = testing_fma(alpha, hx_val[i], hy_gold[hx_ind[i] - idxBase]);
    }

    // Verify results against host
    unit_check_general(1, size, 1, hy_gold.data(), hy.data());

    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpVec(x));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y));

#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_AXPBY_HPP
