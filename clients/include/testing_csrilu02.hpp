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
#ifndef TESTING_CSRILU0_HPP
#define TESTING_CSRILU0_HPP

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
void testing_csrilu02_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVCC__
    // do not test for bad args
    return;
#endif
    int                    m         = 100;
    int                    nnz       = 100;
    int                    safe_size = 100;
    hipsparseSolvePolicy_t policy    = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
    hipsparseStatus_t      status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<csrilu02_struct> unique_ptr_csrilu02(new csrilu02_struct);
    csrilu02Info_t                   info = unique_ptr_csrilu02->info;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*  dptr    = (int*)dptr_managed.get();
    int*  dcol    = (int*)dcol_managed.get();
    T*    dval    = (T*)dval_managed.get();
    void* dbuffer = (void*)dbuffer_managed.get();

    if(!dval || !dptr || !dcol || !dbuffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing hipsparseXcsrilu02_bufferSizeExt
    size_t size;

    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXcsrilu02_bufferSizeExt(
            handle, m, nnz, descr, dval, dptr_null, dcol, info, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXcsrilu02_bufferSizeExt(
            handle, m, nnz, descr, dval, dptr, dcol_null, info, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXcsrilu02_bufferSizeExt(
            handle, m, nnz, descr, dval_null, dptr, dcol, info, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == buffer_size)
    {
        size_t* size_null = nullptr;

        status = hipsparseXcsrilu02_bufferSizeExt(
            handle, m, nnz, descr, dval, dptr, dcol, info, size_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: size is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXcsrilu02_bufferSizeExt(
            handle, m, nnz, descr_null, dval, dptr, dcol, info, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        csrilu02Info_t info_null = nullptr;

        status = hipsparseXcsrilu02_bufferSizeExt(
            handle, m, nnz, descr, dval, dptr, dcol, info_null, &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrilu02_bufferSizeExt(
            handle_null, m, nnz, descr, dval, dptr, dcol, info, &size);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing hipsparseXcsrilu02_analysis

    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr, dval, dptr_null, dcol, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr, dval, dptr, dcol_null, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr, dval_null, dptr, dcol, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        void* dbuffer_null = nullptr;

        status = hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr, dval, dptr, dcol, info, policy, dbuffer_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr_null, dval, dptr, dcol, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        csrilu02Info_t info_null = nullptr;

        status = hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr, dval, dptr, dcol, info_null, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrilu02_analysis(
            handle_null, m, nnz, descr, dval, dptr, dcol, info, policy, dbuffer);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing hipsparseXcsrilu02

    // testing for(nullptr == dptr)
    {
        int* dptr_null = nullptr;

        status = hipsparseXcsrilu02(
            handle, m, nnz, descr, dval, dptr_null, dcol, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        int* dcol_null = nullptr;

        status = hipsparseXcsrilu02(
            handle, m, nnz, descr, dval, dptr, dcol_null, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = hipsparseXcsrilu02(
            handle, m, nnz, descr, dval_null, dptr, dcol, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        void* dbuffer_null = nullptr;

        status = hipsparseXcsrilu02(
            handle, m, nnz, descr, dval, dptr, dcol, info, policy, dbuffer_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
    // testing for(nullptr == descr)
    {
        hipsparseMatDescr_t descr_null = nullptr;

        status = hipsparseXcsrilu02(
            handle, m, nnz, descr_null, dval, dptr, dcol, info, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        csrilu02Info_t info_null = nullptr;

        status = hipsparseXcsrilu02(
            handle, m, nnz, descr, dval, dptr, dcol, info_null, policy, dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrilu02(
            handle_null, m, nnz, descr, dval, dptr, dcol, info, policy, dbuffer);
        verify_hipsparse_status_invalid_handle(status);
    }

    // testing hipsparseXcsrilu02_zeroPivot
    int position;

    // testing for(nullptr == position)
    {
        int* position_null = nullptr;

        status = hipsparseXcsrilu02_zeroPivot(handle, info, position_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: position is nullptr");
    }
    // testing for(nullptr == info)
    {
        csrilu02Info_t info_null = nullptr;

        status = hipsparseXcsrilu02_zeroPivot(handle, info_null, &position);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsrilu02_zeroPivot(handle_null, info, &position);
        verify_hipsparse_status_invalid_handle(status);
    }
}

template <typename T>
hipsparseStatus_t testing_csrilu02(Arguments argus)
{
    int                    safe_size = 100;
    int                    m         = argus.M;
    hipsparseIndexBase_t   idx_base  = argus.idx_base;
    hipsparseSolvePolicy_t policy    = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
    std::string            binfile   = "";
    std::string            filename  = "";
    hipsparseStatus_t      status;
    size_t                 size;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    if(m == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        m       = safe_size;
    }

    if(argus.timing == 1)
    {
        filename = argus.filename;
    }

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr(new descr_struct);
    hipsparseMatDescr_t           descr = test_descr->descr;

    std::unique_ptr<csrilu02_struct> unique_ptr_csrilu02(new csrilu02_struct);
    csrilu02Info_t                   info = unique_ptr_csrilu02->info;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    // Determine number of non-zero elements
    double scale = 0.02;
    if(m > 1000)
    {
        scale = 2.0 / m;
    }
    int nnz = m * scale * m;

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || nnz <= 0)
    {
#ifdef __HIP_PLATFORM_NVCC__
        // Do not test args in cusparse
        return HIPSPARSE_STATUS_SUCCESS;
#endif
        auto dptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dcol_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto buffer_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

        int*  dptr   = (int*)dptr_managed.get();
        int*  dcol   = (int*)dcol_managed.get();
        T*    dval   = (T*)dval_managed.get();
        void* buffer = (void*)buffer_managed.get();

        if(!dval || !dptr || !dcol || !buffer)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!dptr || !dcol || !dval || !buffer");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        // Test hipsparseXcsrilu02_bufferSizeExt
        status = hipsparseXcsrilu02_bufferSizeExt(
            handle, m, nnz, descr, dval, dptr, dcol, info, &size);

        if(m < 0 || nnz < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || nnz < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && nnz >= 0");
        }

        // Test hipsparseXcsrilu02_analysis
        status = hipsparseXcsrilu02_analysis(
            handle, m, nnz, descr, dval, dptr, dcol, info, policy, buffer);

        if(m < 0 || nnz < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || nnz < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && nnz >= 0");
        }

        // Test hipsparseXcsrilu02
        status = hipsparseXcsrilu02(handle, m, nnz, descr, dval, dptr, dcol, info, policy, buffer);

        if(m < 0 || nnz < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || nnz < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && nnz >= 0");
        }

        // Test hipsparseXcsrilu02_zeroPivot
        int zero_pivot;
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrilu02_zeroPivot(handle, info, &zero_pivot));

        // Zero pivot should be -1
        int res = -1;
        unit_check_general(1, 1, 1, &res, &zero_pivot);

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;

    // Initial Data on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(
               binfile.c_str(), m, m, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }
    else if(argus.laplacian)
    {
        m   = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base);
        nnz = hcsr_row_ptr[m];
    }
    else
    {
        std::vector<int> hcoo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(
                   filename.c_str(), m, m, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, idx_base)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            gen_matrix_coo(m, m, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, idx_base);
        }

        // Convert COO to CSR
        hcsr_row_ptr.resize(m + 1, 0);
        for(int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptr[hcoo_row_ind[i] + 1 - idx_base];
        }

        hcsr_row_ptr[0] = idx_base;
        for(int i = 0; i < m; ++i)
        {
            hcsr_row_ptr[i + 1] += hcsr_row_ptr[i];
        }
    }

    // Allocate memory on device
    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto d_position_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};

    int* dptr       = (int*)dptr_managed.get();
    int* dcol       = (int*)dcol_managed.get();
    T*   dval       = (T*)dval_managed.get();
    int* d_position = (int*)d_position_managed.get();

    if(!dval || !dptr || !dcol || !d_position)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dval || !dptr || !dcol || !d_position");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain csrilu02 buffer size
    CHECK_HIPSPARSE_ERROR(
        hipsparseXcsrilu02_bufferSizeExt(handle, m, nnz, descr, dval, dptr, dcol, info, &size));

    // Allocate buffer on the device
    auto dbuffer_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * size), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    if(!dbuffer)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED, "!dbuffer");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // csrilu02 analysis
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrilu02_analysis(
        handle, m, nnz, descr, dval, dptr, dcol, info, policy, dbuffer));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(
            hipsparseXcsrilu02(handle, m, nnz, descr, dval, dptr, dcol, info, policy, dbuffer));

        // Pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        int               hposition_1;
        hipsparseStatus_t pivot_status_1;
        pivot_status_1 = hipsparseXcsrilu02_zeroPivot(handle, info, &hposition_1);

        // Pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));

        hipsparseStatus_t pivot_status_2;
        pivot_status_2 = hipsparseXcsrilu02_zeroPivot(handle, info, d_position);

        // Copy output from device to CPU
        int            hposition_2;
        std::vector<T> result(nnz);
        CHECK_HIP_ERROR(hipMemcpy(result.data(), dval, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(&hposition_2, d_position, sizeof(int), hipMemcpyDeviceToHost));

        // Host csrilu02
        double cpu_time_used = get_time_us();

        int position_gold
            = csrilu0(m, hcsr_row_ptr.data(), hcsr_col_ind.data(), hcsr_val.data(), idx_base);

        cpu_time_used = get_time_us() - cpu_time_used;

        unit_check_general(1, 1, 1, &position_gold, &hposition_1);
        unit_check_general(1, 1, 1, &position_gold, &hposition_2);

        if(hposition_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(pivot_status_1,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
            return HIPSPARSE_STATUS_SUCCESS;
        }

        if(hposition_2 != -1)
        {
            verify_hipsparse_status_zero_pivot(pivot_status_2,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
            return HIPSPARSE_STATUS_SUCCESS;
        }

#if defined(__HIP_PLATFORM_HCC__)
        unit_check_general(1, nnz, 1, hcsr_val.data(), result.data());
#elif defined(__HIP_PLATFORM_NVCC__)
        // do weaker check for cusparse
        unit_check_near(1, nnz, 1, hcsr_val.data(), result.data());
#endif
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSRILU0_HPP
