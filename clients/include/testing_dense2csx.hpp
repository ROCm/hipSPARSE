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
#ifndef TESTING_DENSE2CSX_HPP
#define TESTING_DENSE2CSX_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

#include <iostream>
template <hipsparseDirection_t DIRA, typename T, typename FUNC>
void testing_dense2csx_bad_arg(FUNC& dense2csx)
{

#ifdef __HIP_PLATFORM_NVIDIA__
    // do not test for bad args
    return;
#endif
    static constexpr size_t safe_size = 100;
    static constexpr int    M         = 10;
    static constexpr int    N         = 10;
    static constexpr int    LD        = M;
    hipsparseStatus_t       status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    auto m_csx_val         = hipsparse_unique_ptr{device_malloc(sizeof(T) * 1), device_free};
    auto m_dense_val       = hipsparse_unique_ptr{device_malloc(sizeof(T) * 1), device_free};
    auto m_nnzPerRowColumn = hipsparse_unique_ptr{device_malloc(sizeof(int) * 1), device_free};
    auto m_csx_row_col_ptr = hipsparse_unique_ptr{device_malloc(sizeof(int) * 1), device_free};
    auto m_csx_row_col_ind = hipsparse_unique_ptr{device_malloc(sizeof(int) * 1), device_free};

    T*   d_dense_val       = (T*)m_dense_val.get();
    T*   d_csx_val         = (T*)m_csx_val.get();
    int* d_nnzPerRowColumn = (int*)m_nnzPerRowColumn.get();
    int* d_csx_row_col_ptr = (int*)m_csx_row_col_ptr.get();
    int* d_csx_col_row_ind = (int*)m_csx_row_col_ind.get();

    if(!d_dense_val || !d_nnzPerRowColumn || !d_csx_row_col_ptr || !d_csx_col_row_ind || !d_csx_val)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    //
    // Testing invalid handle.
    //
    status = dense2csx(
        nullptr, 0, 0, nullptr, (const T*)nullptr, 0, nullptr, (T*)nullptr, nullptr, nullptr);
    verify_hipsparse_status_invalid_handle(status);

    //
    // Testing invalid pointers.
    //
    status = dense2csx(handle,
                       M,
                       N,
                       nullptr,
                       d_dense_val,
                       LD,
                       d_nnzPerRowColumn,
                       d_csx_val,
                       d_csx_row_col_ptr,
                       d_csx_col_row_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.");

    status = dense2csx(handle,
                       M,
                       N,
                       descr,
                       nullptr,
                       LD,
                       d_nnzPerRowColumn,
                       d_csx_val,
                       d_csx_row_col_ptr,
                       d_csx_col_row_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.");
    status = dense2csx(handle,
                       M,
                       N,
                       descr,
                       d_dense_val,
                       LD,
                       nullptr,
                       d_csx_val,
                       d_csx_row_col_ptr,
                       d_csx_col_row_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.");

    status = dense2csx(handle,
                       M,
                       N,
                       descr,
                       d_dense_val,
                       LD,
                       d_nnzPerRowColumn,
                       nullptr,
                       d_csx_row_col_ptr,
                       d_csx_col_row_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.");
    status = dense2csx(handle,
                       M,
                       N,
                       descr,
                       d_dense_val,
                       LD,
                       d_nnzPerRowColumn,
                       d_csx_val,
                       nullptr,
                       d_csx_col_row_ind);

    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.");
    status = dense2csx(handle,
                       M,
                       N,
                       descr,
                       d_dense_val,
                       LD,
                       d_nnzPerRowColumn,
                       d_csx_val,
                       d_csx_row_col_ptr,
                       nullptr);

    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.");

    //
    // Testing invalid size on M
    //
    status = dense2csx(handle,
                       -1,
                       N,
                       descr,
                       d_dense_val,
                       LD,
                       d_nnzPerRowColumn,
                       d_csx_val,
                       d_csx_row_col_ptr,
                       d_csx_col_row_ind);
    verify_hipsparse_status_invalid_size(status, "Error: An invalid size must be detected.");
    //
    // Testing invalid size on N
    //
    status = dense2csx(handle,
                       M,
                       -1,
                       descr,
                       d_dense_val,
                       LD,
                       d_nnzPerRowColumn,
                       d_csx_val,
                       d_csx_row_col_ptr,
                       d_csx_col_row_ind);
    verify_hipsparse_status_invalid_size(status, "Error: An invalid size must be detected.");
    //
    // Testing invalid size on LD
    //
    status = dense2csx(handle,
                       M,
                       N,
                       descr,
                       d_dense_val,
                       M - 1,
                       d_nnzPerRowColumn,
                       d_csx_val,
                       d_csx_row_col_ptr,
                       d_csx_col_row_ind);
    verify_hipsparse_status_invalid_size(status, "Error: An invalid size must be detected.");
}

template <hipsparseDirection_t DIRA, typename T, typename FUNC>
hipsparseStatus_t testing_dense2csx(const Arguments& argus, FUNC& dense2csx)
{

    int                  M        = argus.M;
    int                  N        = argus.N;
    int                  LD       = argus.lda;
    hipsparseIndexBase_t idx_base = argus.idx_base;

    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));

    if(M <= 0 || N <= 0 || LD < M)
    {
        hipsparseStatus_t expected_status
            = (((M == 0 && N >= 0) || (M >= 0 && N == 0)) && (LD >= M))
                  ? HIPSPARSE_STATUS_SUCCESS
                  : HIPSPARSE_STATUS_INVALID_VALUE;
        status = dense2csx(
            handle, M, N, descr, (const T*)nullptr, LD, nullptr, (T*)nullptr, nullptr, nullptr);
        verify_hipsparse_status(status,
                                expected_status,
                                (expected_status == HIPSPARSE_STATUS_SUCCESS)
                                    ? "Error: call with zero sizes must be succesfull."
                                    : "Error: An invalid size must be detected.");

        return HIPSPARSE_STATUS_SUCCESS;
    }

    int              DIMDIR = (HIPSPARSE_DIRECTION_ROW == DIRA) ? M : N;
    std::vector<T>   h_dense_val(LD * N);
    std::vector<int> h_nnzPerRowColumn(DIMDIR);
    std::vector<int> hd_nnzPerRowColumn(DIMDIR);
    std::vector<int> h_nnzTotalDevHostPtr(1);
    std::vector<int> hd_nnzTotalDevHostPtr(1);

    //
    // Create the dense matrix.
    //
    int  MN          = DIMDIR;
    auto m_dense_val = hipsparse_unique_ptr{device_malloc(sizeof(T) * LD * N), device_free};
    auto nnzPerRowColumn_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * MN), device_free};
    auto nnzTotalDevHostPtr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * 1), device_free};

    T*   d_dense_val          = (T*)m_dense_val.get();
    int* d_nnzPerRowColumn    = (int*)nnzPerRowColumn_managed.get();
    int* d_nnzTotalDevHostPtr = (int*)nnzTotalDevHostPtr_managed.get();
    if(!d_nnzPerRowColumn || !d_nnzTotalDevHostPtr || !d_dense_val)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!h_nnzPerRowColumn || !d_nnzPerRowColumn || !d_dense_val");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    //
    // Initialize the entire allocated memory.
    //
    for(int i = 0; i < LD; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            h_dense_val[j * LD + i] = make_DataType<T>(-1);
        }
    }

    //
    // Initialize a random dense matrix.
    //
    srand(0);
    gen_dense_random_sparsity_pattern(M, N, h_dense_val.data(), LD, 0.2);

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(
        hipMemcpy(d_dense_val, h_dense_val.data(), sizeof(T) * LD * N, hipMemcpyHostToDevice));

    int nnz;
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(
        hipsparseXnnz(handle, DIRA, M, N, descr, d_dense_val, LD, d_nnzPerRowColumn, &nnz));

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(hipMemcpy(
        h_nnzPerRowColumn.data(), d_nnzPerRowColumn, sizeof(int) * DIMDIR, hipMemcpyDeviceToHost));

    auto m_csx_row_col_ptr
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (DIMDIR + 1)), device_free};
    auto m_csx_val         = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto m_csx_col_row_ind = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};

    int* d_csx_row_col_ptr = (int*)m_csx_row_col_ptr.get();
    int* d_csx_col_row_ind = (int*)m_csx_col_row_ind.get();
    T*   d_csx_val         = (T*)m_csx_val.get();

    if(!d_csx_row_col_ptr || !d_csx_val || !d_csx_col_row_ind)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    std::vector<int> cpu_csx_row_col_ptr(DIMDIR + 1);
    std::vector<T>   cpu_csx_val(nnz);
    std::vector<int> cpu_csx_col_row_ind(nnz);
    if(!cpu_csx_row_col_ptr.data() || !cpu_csx_val.data() || !cpu_csx_col_row_ind.data())
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    if(argus.unit_check)
    {

        //
        // Compute the reference host first.
        //

        host_dense2csx<DIRA, T>(M,
                                N,
                                hipsparseGetMatIndexBase(descr),
                                h_dense_val.data(),
                                LD,
                                h_nnzPerRowColumn.data(),
                                cpu_csx_val.data(),
                                cpu_csx_row_col_ptr.data(),
                                cpu_csx_col_row_ind.data());

        if(DIRA == HIPSPARSE_DIRECTION_ROW)
        {
            CHECK_HIPSPARSE_ERROR(dense2csx(handle,
                                            M,
                                            N,
                                            descr,
                                            d_dense_val,
                                            LD,
                                            d_nnzPerRowColumn,
                                            d_csx_val,
                                            d_csx_row_col_ptr,
                                            d_csx_col_row_ind));
        }
        else
        {
            CHECK_HIPSPARSE_ERROR(dense2csx(handle,
                                            M,
                                            N,
                                            descr,
                                            d_dense_val,
                                            LD,
                                            d_nnzPerRowColumn,
                                            d_csx_val,
                                            d_csx_col_row_ind,
                                            d_csx_row_col_ptr));
        }

        void* buffer = malloc(std::max(sizeof(T), sizeof(int)) * std::max(DIMDIR + 1, nnz));
        //
        // Transfer and check results.
        //
        CHECK_HIP_ERROR(hipMemcpy(buffer, d_csx_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        unit_check_general(1, nnz, 1, cpu_csx_val.data(), (T*)buffer);
        CHECK_HIP_ERROR(
            hipMemcpy(buffer, d_csx_col_row_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));
        unit_check_general(1, nnz, 1, cpu_csx_col_row_ind.data(), (int*)buffer);

        CHECK_HIP_ERROR(hipMemcpy(
            buffer, d_csx_row_col_ptr, sizeof(int) * (DIMDIR + 1), hipMemcpyDeviceToHost));

        unit_check_general(1, (DIMDIR + 1), 1, cpu_csx_row_col_ptr.data(), (int*)buffer);

        free(buffer);
        buffer = nullptr;
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_DENSE2CSX_HPP
