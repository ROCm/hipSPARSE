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
#ifndef TESTING_SPVV_HPP
#define TESTING_SPVV_HPP

#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <typeinfo>

using namespace hipsparse_test;

void testing_spvv_bad_arg(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
    int64_t size = 100;
    int64_t nnz  = 100;

    float result;

    hipsparseOperation_t opType   = HIPSPARSE_OPERATION_NON_TRANSPOSE;
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

    // SpVV bufferSize
    size_t bufferSize;
    verify_hipsparse_status_invalid_handle(
        hipsparseSpVV_bufferSize(nullptr, opType, x, y, &result, dataType, &bufferSize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV_bufferSize(handle, opType, nullptr, y, &result, dataType, &bufferSize),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV_bufferSize(handle, opType, x, nullptr, &result, dataType, &bufferSize),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV_bufferSize(handle, opType, x, y, nullptr, dataType, &bufferSize),
        "Error: result is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV_bufferSize(handle, opType, x, y, &result, dataType, nullptr),
        "Error: bufferSize is nullptr");

    // SpVV
    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, 100));

    verify_hipsparse_status_invalid_handle(
        hipsparseSpVV(nullptr, opType, x, y, &result, dataType, buffer));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV(handle, opType, nullptr, y, &result, dataType, buffer),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV(handle, opType, x, nullptr, &result, dataType, buffer),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV(handle, opType, x, y, nullptr, dataType, buffer), "Error: result is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV(handle, opType, x, y, &result, dataType, nullptr),
        "Error: buffer is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpVec(x), "Success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(y), "Success");

    CHECK_HIP_ERROR(hipFree(buffer));
#endif
}

template <typename I, typename T>
hipsparseStatus_t testing_spvv(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
    int64_t size = 15332;
    int64_t nnz  = 500;

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

    T hresult_N;
    T hresult_C;
    T hresult_N_gold;
    T hresult_C_gold;

    // Initial Data on CPU
    srand(12345ULL);
    hipsparseInitIndex(hx_ind.data(), nnz, 1, size);
    hipsparseInit<T>(hx_val, 1, nnz);
    hipsparseInit<T>(hy, 1, size);

    // Allocate memory on device
    auto dx_ind_managed    = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dx_val_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_managed        = hipsparse_unique_ptr{device_malloc(sizeof(T) * size), device_free};
    auto dresult_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dx_ind    = (I*)dx_ind_managed.get();
    T* dx_val    = (T*)dx_val_managed.get();
    T* dy        = (T*)dy_managed.get();
    T* dresult_C = (T*)dresult_C_managed.get();

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

    // SpVV_bufferSize
    size_t bufferSize;
    void*  externalBuffer;

    // SpVV non-transpose pointer-mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseSpVV_bufferSize(
        handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, x, y, &hresult_N, dataType, &bufferSize));
    CHECK_HIP_ERROR(hipMalloc(&externalBuffer, bufferSize));
    CHECK_HIPSPARSE_ERROR(hipsparseSpVV(
        handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, x, y, &hresult_N, dataType, externalBuffer));
    CHECK_HIP_ERROR(hipFree(externalBuffer));

    // SpVV conjugate-transpose pointer-mode device
    if(dataType == HIP_C_32F || dataType == HIP_C_64F)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseSpVV_bufferSize(handle,
                                                       HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                                                       x,
                                                       y,
                                                       dresult_C,
                                                       dataType,
                                                       &bufferSize));
        CHECK_HIP_ERROR(hipMalloc(&externalBuffer, bufferSize));
        CHECK_HIPSPARSE_ERROR(hipsparseSpVV(handle,
                                            HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                                            x,
                                            y,
                                            dresult_C,
                                            dataType,
                                            externalBuffer));
        CHECK_HIP_ERROR(hipFree(externalBuffer));

        // Copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(&hresult_C, dresult_C, sizeof(T), hipMemcpyDeviceToHost));
    }

    // CPU non-transpose
    hresult_N_gold = make_DataType<T>(0);
    for(I i = 0; i < nnz; ++i)
    {
        hresult_N_gold = hresult_N_gold + testing_mult(hy[hx_ind[i] - idxBase], hx_val[i]);
    }

    // Verify results against host
    unit_check_general(1, 1, 1, &hresult_N_gold, &hresult_N);

    // CPU transpose
    if(dataType == HIP_C_32F || dataType == HIP_C_64F)
    {
        hresult_C_gold = make_DataType<T>(0);
        for(I i = 0; i < nnz; ++i)
        {
            hresult_C_gold
                = hresult_C_gold + testing_mult(testing_conj(hx_val[i]), hy[hx_ind[i] - idxBase]);
        }

        // Verify results against host
        unit_check_general(1, 1, 1, &hresult_C_gold, &hresult_C);
    }

    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpVec(x));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y));
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_SPVV_HPP

