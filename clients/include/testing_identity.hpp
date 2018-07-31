/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_IDENTITY_HPP
#define TESTING_IDENTITY_HPP

#include "hipsparse_test_unique_ptr.hpp"
#include "hipsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <hipsparse.h>
#include <algorithm>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_identity_bad_arg(void)
{
    int n         = 100;
    int safe_size = 100;
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t handle = unique_ptr_handle->handle;

    auto p_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};

    int* p = (int*)p_managed.get();

    if(!p)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing for (p == nullptr)
    {
        int* p_null = nullptr;

        status = hipsparseCreateIdentityPermutation(handle, n, p_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: p is nullptr");
    }

    // Testing for(handle == nullptr)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseCreateIdentityPermutation(handle_null, n, p);
        verify_hipsparse_status_invalid_handle(status);
    }
}

hipsparseStatus_t testing_identity(Arguments argus)
{
    int n         = argus.N;
    int safe_size = 100;
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t handle = unique_ptr_handle->handle;

    // Argument sanity check before allocating invalid memory
    if(n <= 0)
    {
        auto p_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};

        int* p = (int*)p_managed.get();

        if(!p)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED, "!p");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        status = hipsparseCreateIdentityPermutation(handle, n, p);

        if(n < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: n < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "n >= 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Host structures
    std::vector<int> hp(n);
    std::vector<int> hp_gold(n);

    // create_identity_permutation on host
    for(int i = 0; i < n; ++i)
    {
        hp_gold[i] = i;
    }

    // Allocate memory on the device
    auto dp_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * n), device_free};

    int* dp = (int*)dp_managed.get();

    if(!dp)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED, "!p");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, n, dp));

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(hp.data(), dp, sizeof(int) * n, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, n, 1, hp_gold.data(), hp.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            hipsparseCreateIdentityPermutation(handle, n, dp);
        }

        double gpu_time_used = get_time_us();

        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            hipsparseCreateIdentityPermutation(handle, n, dp);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        double bandwidth = sizeof(int) * n / gpu_time_used / 1e6;

        printf("n\t\tGB/s\tmsec\n");
        printf("%8d\t%0.2lf\t%0.2lf\n", n, bandwidth, gpu_time_used);
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_IDENTITY_HPP
