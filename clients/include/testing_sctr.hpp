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
#ifndef TESTING_SCTR_HPP
#define TESTING_SCTR_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_sctr_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVCC__
    // do not test for bad args
    return;
#endif
    int nnz       = 100;
    int safe_size = 100;

    hipsparseIndexBase_t idx_base = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseStatus_t    status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dx_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dx_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    T*   dx_val = (T*)dx_val_managed.get();
    int* dx_ind = (int*)dx_ind_managed.get();
    T*   dy     = (T*)dy_managed.get();

    if(!dx_ind || !dx_val || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing for(nullptr == dx_ind)
    {
        int* dx_ind_null = nullptr;

        status = hipsparseXsctr(handle, nnz, dx_val, dx_ind_null, dy, idx_base);
        verify_hipsparse_status_invalid_pointer(status, "Error: x_ind is nullptr");
    }
    // testing for(nullptr == dx_val)
    {
        T* dx_val_null = nullptr;

        status = hipsparseXsctr(handle, nnz, dx_val_null, dx_ind, dy, idx_base);
        verify_hipsparse_status_invalid_pointer(status, "Error: x_val is nullptr");
    }
    // testing for(nullptr == dy)
    {
        T* dy_null = nullptr;

        status = hipsparseXsctr(handle, nnz, dx_val, dx_ind, dy_null, idx_base);
        verify_hipsparse_status_invalid_pointer(status, "Error: y is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXsctr(handle_null, nnz, dx_val, dx_ind, dy, idx_base);
        verify_hipsparse_status_invalid_handle(status);
    }
}

template <typename T>
hipsparseStatus_t testing_sctr(Arguments argus)
{
    int                  N         = argus.N;
    int                  nnz       = argus.nnz;
    int                  safe_size = 100;
    hipsparseIndexBase_t idx_base  = argus.idx_base;
    hipsparseStatus_t    status;

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    // Argument sanity check before allocating invalid memory
    if(nnz <= 0)
    {
#ifdef __HIP_PLATFORM_NVCC__
        // Do not test args in cusparse
        return HIPSPARSE_STATUS_SUCCESS;
#endif
        auto dx_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dx_val_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dy_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* dx_ind = (int*)dx_ind_managed.get();
        T*   dx_val = (T*)dx_val_managed.get();
        T*   dy     = (T*)dy_managed.get();

        if(!dx_ind || !dx_val || !dy)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!dx_ind || !dx_val || !dy");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        status = hipsparseXsctr(handle, nnz, dx_val, dx_ind, dy, idx_base);

        if(nnz < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: nnz < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "nnz == 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Host structures
    std::vector<int> hx_ind(nnz);
    std::vector<T>   hx_val(nnz);
    std::vector<T>   hy(N);
    std::vector<T>   hy_gold(N);

    // Initial Data on CPU
    srand(12345ULL);
    hipsparseInitIndex(hx_ind.data(), nnz, 1, N);
    hipsparseInit<T>(hx_val, 1, nnz);
    hipsparseInit<T>(hy, 1, N);

    hy_gold = hy;

    // allocate memory on device
    auto dx_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dx_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * N), device_free};

    int* dx_ind = (int*)dx_ind_managed.get();
    T*   dx_val = (T*)dx_val_managed.get();
    T*   dy     = (T*)dy_managed.get();

    if(!dx_ind || !dx_val || !dy)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED, "!dx_ind || !dx_val || !dy");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * N, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        // ROCSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXsctr(handle, nnz, dx_val, dx_ind, dy, idx_base));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * N, hipMemcpyDeviceToHost));

        // CPU
        double cpu_time_used = get_time_us();

        for(int i = 0; i < nnz; ++i)
        {
            hy_gold[hx_ind[i] - idx_base] = hx_val[i];
        }

        cpu_time_used = get_time_us() - cpu_time_used;

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        unit_check_general(1, N, 1, hy_gold.data(), hy.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            hipsparseXsctr(handle, nnz, dx_val, dx_ind, dy, idx_base);
        }

        double gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            hipsparseXsctr(handle, nnz, dx_val, dx_ind, dy, idx_base);
        }

        gpu_time_used    = (get_time_us() - gpu_time_used) / number_hot_calls;
        double bandwidth = (sizeof(int) * nnz + sizeof(T) * 2.0 * nnz) / gpu_time_used / 1e3;

        printf("nnz\t\tGB/s\tusec\n");
        printf("%9d\t%0.2lf\t%0.2lf\n", nnz, bandwidth, gpu_time_used);
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_SCTR_HPP
