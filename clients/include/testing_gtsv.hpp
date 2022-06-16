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
#ifndef TESTING_GTSV2_HPP
#define TESTING_GTSV2_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_gtsv2_bad_arg(void)
{
    // Dont do bad argument checking for cuda
#if(!defined(CUDART_VERSION))
    int safe_size = 100;
    int m         = 10;
    int n         = 10;
    int ldb       = m;

    // Create handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto ddl_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dd_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto ddu_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    T*    ddl  = (T*)ddl_managed.get();
    T*    dd   = (T*)dd_managed.get();
    T*    ddu  = (T*)ddu_managed.get();
    T*    dB   = (T*)dB_managed.get();
    void* dbuf = (void*)dbuf_managed.get();

    if(!ddl || !dd || !ddu || !dB || !dbuf)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    size_t bsize;

    // gtsv buffer size
    verify_hipsparse_status_invalid_handle(
        hipsparseXgtsv2_bufferSizeExt(nullptr, m, n, ddl, dd, ddu, dB, ldb, &bsize));
    verify_hipsparse_status_invalid_value(
        hipsparseXgtsv2_bufferSizeExt(handle, -1, n, ddl, dd, ddu, dB, ldb, &bsize),
        "Error: m is invalid");
    verify_hipsparse_status_invalid_value(
        hipsparseXgtsv2_bufferSizeExt(handle, m, -1, ddl, dd, ddu, dB, ldb, &bsize),
        "Error: n is invalid");
    verify_hipsparse_status_invalid_value(
        hipsparseXgtsv2_bufferSizeExt(handle, m, n, ddl, dd, ddu, dB, -1, &bsize),
        "Error: ldb is invalid");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgtsv2_bufferSizeExt(handle, m, n, ddl, dd, ddu, dB, ldb, nullptr),
        "Error: bsize is nullptr");

    // gtsv
    verify_hipsparse_status_invalid_handle(
        hipsparseXgtsv2(nullptr, m, n, ddl, dd, ddu, dB, ldb, dbuf));
    verify_hipsparse_status_invalid_value(
        hipsparseXgtsv2(handle, -1, n, ddl, dd, ddu, dB, ldb, dbuf), "Error: m is invalid");
    verify_hipsparse_status_invalid_value(
        hipsparseXgtsv2(handle, m, -1, ddl, dd, ddu, dB, ldb, dbuf), "Error: n is invalid");
    verify_hipsparse_status_invalid_value(hipsparseXgtsv2(handle, m, n, ddl, dd, ddu, dB, -1, dbuf),
                                          "Error: ldb is invalid");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgtsv2(handle, m, n, (const T*)nullptr, dd, ddu, dB, ldb, dbuf),
        "Error: ddl is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgtsv2(handle, m, n, ddl, (const T*)nullptr, ddu, dB, ldb, dbuf),
        "Error: dd is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgtsv2(handle, m, n, ddl, dd, (const T*)nullptr, dB, ldb, dbuf),
        "Error: ddu is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgtsv2(handle, m, n, ddl, dd, ddu, (T*)nullptr, ldb, dbuf),
        "Error: dB is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgtsv2(handle, m, n, ddl, dd, ddu, dB, ldb, nullptr), "Error: bsize is nullptr");
#endif
}

template <typename T>
hipsparseStatus_t testing_gtsv2(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)

    // hipSPARSE handle
    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    int m   = 512;
    int n   = 512;
    int ldb = 2 * m;

    // Host structures
    std::vector<T> hdl(m, make_DataType<T>(1));
    std::vector<T> hd(m, make_DataType<T>(2));
    std::vector<T> hdu(m, make_DataType<T>(1));
    std::vector<T> hB(ldb * n, make_DataType<T>(3));

    hdl[0]     = make_DataType<T>(0);
    hdu[m - 1] = make_DataType<T>(0);

    std::vector<T> hB_original = hB;

    // allocate memory on device
    auto ddl_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dd_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto ddu_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dB_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T) * ldb * n), device_free};

    T* ddl = (T*)ddl_managed.get();
    T* dd  = (T*)dd_managed.get();
    T* ddu = (T*)ddu_managed.get();
    T* dB  = (T*)dB_managed.get();

    if(!ddl || !dd || !ddu || !dB)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!ddl || !dd || !ddu || !dB");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(ddl, hdl.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dd, hd.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ddu, hdu.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * ldb * n, hipMemcpyHostToDevice));

    // Query SparseToDense buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(
        hipsparseXgtsv2_bufferSizeExt(handle, m, n, ddl, dd, ddu, dB, ldb, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    CHECK_HIPSPARSE_ERROR(hipsparseXgtsv2(handle, m, n, ddl, dd, ddu, dB, ldb, buffer));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hB.data(), dB, sizeof(T) * ldb * n, hipMemcpyDeviceToHost));

    // Check
    std::vector<T> hresult = hB_original;
    for(int j = 0; j < n; j++)
    {
        hresult[ldb * j] = testing_mult(hd[0], hB[ldb * j]) + testing_mult(hdu[0], hB[ldb * j + 1]);
        hresult[ldb * j + m - 1]
            = testing_mult(hdl[m - 1], hB[ldb * j + m - 2]) + testing_mult(hd[m - 1], hB[ldb * j + m - 1]);
        for(int i = 1; i < m - 1; i++)
        {
            hresult[ldb * j + i] = testing_mult(hdl[i], hB[ldb * j + i - 1]) + testing_mult(hd[i], hB[ldb * j + i])
                                   + testing_mult(hdu[i], hB[ldb * j + i + 1]);
        }
    }

    unit_check_near(m, n, ldb, hB_original.data(), hresult.data());

    CHECK_HIP_ERROR(hipFree(buffer));
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_GTSV2_HPP
