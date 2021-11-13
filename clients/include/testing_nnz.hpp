/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef TESTING_NNZ_HPP
#define TESTING_NNZ_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse/hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_nnz_bad_arg(void)
{
    static constexpr size_t               safe_size = 100;
    static constexpr int                  M         = 10;
    static constexpr int                  N         = 10;
    static constexpr int                  lda       = M;
    static constexpr hipsparseDirection_t dirA      = HIPSPARSE_DIRECTION_ROW;
    hipsparseStatus_t                     status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descrA = unique_ptr_descr->descr;

    auto A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto nnzPerRowColumn_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto nnzTotalDevHostPtr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * 1), device_free};

    T*   d_A                  = (T*)A_managed.get();
    int* d_nnzPerRowColumn    = (int*)nnzPerRowColumn_managed.get();
    int* d_nnzTotalDevHostPtr = (int*)nnzTotalDevHostPtr_managed.get();

    if(!d_nnzPerRowColumn || !d_A || !d_nnzTotalDevHostPtr)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

#if(!defined(CUDART_VERSION))
    //
    // Testing invalid handle.
    //
    status = hipsparseXnnz(
        nullptr, dirA, M, N, descrA, (const T*)d_A, lda, d_nnzPerRowColumn, d_nnzTotalDevHostPtr);
    verify_hipsparse_status_invalid_handle(status);

    //
    // Testing invalid pointers.
    //
    status = hipsparseXnnz(
        handle, dirA, M, N, nullptr, (const T*)d_A, lda, d_nnzPerRowColumn, d_nnzTotalDevHostPtr);

    verify_hipsparse_status_invalid_pointer(status,
                                            "Error: descrA as invalid pointer must be detected.");

    status = hipsparseXnnz(handle,
                           dirA,
                           M,
                           N,
                           descrA,
                           (const T*)nullptr,
                           lda,
                           d_nnzPerRowColumn,
                           d_nnzTotalDevHostPtr);
    verify_hipsparse_status_invalid_pointer(status,
                                            "Error: A as invalid pointer must be detected.");

    status = hipsparseXnnz(
        handle, dirA, M, N, descrA, (const T*)d_A, lda, nullptr, d_nnzTotalDevHostPtr);
    verify_hipsparse_status_invalid_pointer(
        status, "Error: nnzPerRowColumn as invalid pointer must be detected.");

    status
        = hipsparseXnnz(handle, dirA, M, N, descrA, (const T*)d_A, lda, d_nnzPerRowColumn, nullptr);
    verify_hipsparse_status_invalid_pointer(
        status, "Error: nnzTotalDevHostPtr as invalid pointer must be detected.");
#endif

    //
    // Testing invalid direction
    //
    try
    {
        status = hipsparseXnnz(handle,
                               (hipsparseDirection_t)77,
                               -1,
                               -1,
                               descrA,
                               (const T*)nullptr,
                               -1,
                               nullptr,
                               nullptr);
        //
        // An exception should be thrown.
        //
        verify_hipsparse_status_internal_error(
            HIPSPARSE_STATUS_SUCCESS,
            "Error: an exception must be thrown from the conversion of the hipsparseDirection_t.");
    }
    catch(...)
    {
    }

    //
    // Testing invalid size on M
    //
    status = hipsparseXnnz(
        handle, dirA, -1, N, descrA, (const T*)d_A, lda, d_nnzPerRowColumn, d_nnzTotalDevHostPtr);
    verify_hipsparse_status_invalid_size(status, "Error: M < 0 must be detected.");

    //
    // Testing invalid size on N
    //
    status = hipsparseXnnz(
        handle, dirA, M, -1, descrA, (const T*)d_A, lda, d_nnzPerRowColumn, d_nnzTotalDevHostPtr);
    verify_hipsparse_status_invalid_size(status, "Error: N < 0 must be detected.");

    //
    // Testing invalid size on lda
    //
    status = hipsparseXnnz(
        handle, dirA, M, N, descrA, (const T*)d_A, M - 1, d_nnzPerRowColumn, d_nnzTotalDevHostPtr);
    verify_hipsparse_status_invalid_size(status, "Error: lda < M must be detected.");
}

template <typename T>
hipsparseStatus_t testing_nnz(Arguments argus)
{
    int                  M    = argus.M;
    int                  N    = argus.N;
    int                  lda  = argus.lda;
    hipsparseDirection_t dirA = argus.dirA;

    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descrA = unique_ptr_descr->descr;

    if(M <= 0 || N <= 0 || lda < M)
    {
	// cusparse returns internal error for this case
#if(defined(CUDART_VERSION))
	if((M == 0 || N == 0) && lda >= M)
	{
	    return HIPSPARSE_STATUS_SUCCESS;
	}
#endif

        status
            = hipsparseXnnz(handle, dirA, M, N, descrA, (const T*)nullptr, lda, nullptr, nullptr);
	if(((M == 0 && N >= 0) || (M >= 0 && N == 0)) && (lda >= M))
        {
            verify_hipsparse_status_success(status, "Error: M or N = 0 must be successful.");
        }
        else
        {
            verify_hipsparse_status_invalid_size(
                status, "Error: M is negative or N is negative or lda < M must be detected.");
        }

        if(HIPSPARSE_STATUS_SUCCESS == status)
        {

            int h_nnz = 77;
            CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

            status = hipsparseXnnz(
                handle, dirA, M, N, descrA, (const T*)nullptr, lda, nullptr, nullptr);
            verify_hipsparse_status_success(status, "Error: M or N = 0 must be successful.");

            status = hipsparseXnnz(
                handle, dirA, M, N, descrA, (const T*)nullptr, lda, nullptr, &h_nnz);
            verify_hipsparse_status_success(status, "Error: M or N = 0 must be successful.");

            if(0 != h_nnz)
            {
                verify_hipsparse_status_success(
                    status,
                    "Error: h_nnz must be zero with a non-null pointer and the pointer mode host.");
            }

            h_nnz            = 139;
            auto nnz_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * 1), device_free};
            int* d_nnz       = (int*)nnz_managed.get();

            CHECK_HIP_ERROR(hipMemcpy((int*)d_nnz, &h_nnz, sizeof(int) * 1, hipMemcpyHostToDevice));

            CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
            status = hipsparseXnnz(
                handle, dirA, M, N, descrA, (const T*)nullptr, lda, nullptr, nullptr);
            verify_hipsparse_status_success(status, "Error: M or N = 0 must be successful.");

            status = hipsparseXnnz(
                handle, dirA, M, N, descrA, (const T*)nullptr, lda, nullptr, (int*)d_nnz);
            verify_hipsparse_status_success(status, "Error: M or N = 0 must be successful.");

            CHECK_HIP_ERROR(hipMemcpy(&h_nnz, (int*)d_nnz, sizeof(int) * 1, hipMemcpyDeviceToHost));

            status = (0 == h_nnz) ? HIPSPARSE_STATUS_SUCCESS : HIPSPARSE_STATUS_INTERNAL_ERROR;

            if(0 != h_nnz)
            {
                verify_hipsparse_status_success(
                    status, "Error: h_nnz must be zero with the pointer mode device.");
            }
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    //
    // Create the dense matrix.
    //
    int  MN        = (dirA == HIPSPARSE_DIRECTION_ROW) ? M : N;
    auto A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * lda * N), device_free};
    auto nnzPerRowColumn_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * MN), device_free};
    auto nnzTotalDevHostPtr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * 1), device_free};

    T*   d_A                  = (T*)A_managed.get();
    int* d_nnzPerRowColumn    = (int*)nnzPerRowColumn_managed.get();
    int* d_nnzTotalDevHostPtr = (int*)nnzTotalDevHostPtr_managed.get();
    if(!d_nnzPerRowColumn || !d_nnzTotalDevHostPtr || !d_A)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!h_nnzPerRowColumn || !d_nnzPerRowColumn || !d_A");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    std::vector<T>   h_A(lda * N);
    std::vector<int> h_nnzPerRowColumn(MN);
    std::vector<int> hd_nnzPerRowColumn(MN);
    std::vector<int> h_nnzTotalDevHostPtr(1);
    std::vector<int> hd_nnzTotalDevHostPtr(1);

    //
    // Initialize the entire allocated memory.
    //
    for(int i = 0; i < lda; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            h_A[j * lda + i] = make_DataType<T>(-1);
        }
    }

    //
    // Initialize a random dense matrix.
    //
    srand(0);
    gen_dense_random_sparsity_pattern(M, N, h_A.data(), lda, 0.2);

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), sizeof(T) * lda * N, hipMemcpyHostToDevice));

    //
    // Unit check.
    //
    if(argus.unit_check)
    {
        //
        // Compute the reference host first.
        //
        host_nnz<T>(dirA,
                    M,
                    N,
                    descrA,
                    h_A.data(),
                    lda,
                    h_nnzPerRowColumn.data(),
                    h_nnzTotalDevHostPtr.data());

        //
        // Pointer mode device for nnz and call.
        //
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXnnz(handle,
                                            dirA,
                                            M,
                                            N,
                                            descrA,
                                            (const T*)d_A,
                                            lda,
                                            d_nnzPerRowColumn,
                                            d_nnzTotalDevHostPtr));

        //
        // Transfer.
        //
        CHECK_HIP_ERROR(hipMemcpy(
            hd_nnzPerRowColumn.data(), d_nnzPerRowColumn, sizeof(int) * MN, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hd_nnzTotalDevHostPtr.data(),
                                  d_nnzTotalDevHostPtr,
                                  sizeof(int) * 1,
                                  hipMemcpyDeviceToHost));

        //
        // Check results.
        //
        unit_check_general<int>(1, MN, 1, hd_nnzPerRowColumn.data(), h_nnzPerRowColumn.data());
        unit_check_general<int>(1, 1, 1, hd_nnzTotalDevHostPtr.data(), h_nnzTotalDevHostPtr.data());

        //
        // Pointer mode host for nnz and call.
        //
        int dh_nnz;
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXnnz(
            handle, dirA, M, N, descrA, (const T*)d_A, lda, d_nnzPerRowColumn, &dh_nnz));

        //
        // Transfer.
        //
        CHECK_HIP_ERROR(hipMemcpy(
            hd_nnzPerRowColumn.data(), d_nnzPerRowColumn, sizeof(int) * MN, hipMemcpyDeviceToHost));

        //
        // Check results.
        //
        unit_check_general<int>(1, MN, 1, hd_nnzPerRowColumn.data(), h_nnzPerRowColumn.data());
        unit_check_general<int>(1, 1, 1, &dh_nnz, h_nnzTotalDevHostPtr.data());
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_NNZ_HPP
