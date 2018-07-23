/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_CSR2COO_HPP
#define TESTING_CSR2COO_HPP

#include "hipsparse_test_unique_ptr.hpp"
#include "hipsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <hipsparse.h>
#include <algorithm>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_csr2coo_bad_arg(void)
{
    int m         = 100;
    int nnz       = 100;
    int safe_size = 100;
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t handle = unique_ptr_handle->handle;

    auto csr_row_ptr_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto coo_row_ind_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};

    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* coo_row_ind = (int*)coo_row_ind_managed.get();

    if(!csr_row_ptr || !coo_row_ind)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing for(csr_row_ptr == nullptr)
    {
        int* csr_row_ptr_null = nullptr;

        status = hipsparseXcsr2coo(
            handle, csr_row_ptr_null, nnz, m, coo_row_ind, HIPSPARSE_INDEX_BASE_ZERO);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }
    // Testing for(coo_row_ind == nullptr)
    {
        int* coo_row_ind_null = nullptr;

        status = hipsparseXcsr2coo(
            handle, csr_row_ptr, nnz, m, coo_row_ind_null, HIPSPARSE_INDEX_BASE_ZERO);
        verify_hipsparse_status_invalid_pointer(status, "Error: coo_row_ind is nullptr");
    }
    // Testing for(handle == nullptr)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsr2coo(
            handle_null, csr_row_ptr, nnz, m, coo_row_ind, HIPSPARSE_INDEX_BASE_ZERO);
        verify_hipsparse_status_invalid_handle(status);
    }
}

hipsparseStatus_t testing_csr2coo(Arguments argus)
{
    int m               = argus.M;
    int n               = argus.N;
    int safe_size       = 100;
    hipsparseIndexBase_t idx_base = argus.idx_base;
    hipsparseStatus_t status;

    double scale = 0.02;
    if(m > 1000 || n > 1000)
    {
        scale = 2.0 / std::max(m, n);
    }
    int nnz = m * scale * n;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t handle = unique_ptr_handle->handle;

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0)
    {
        auto csr_row_ptr_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto coo_row_ind_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};

        int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
        int* coo_row_ind = (int*)coo_row_ind_managed.get();

        if(!csr_row_ptr || !coo_row_ind)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!csr_row_ptr || !coo_row_ind");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        status = hipsparseXcsr2coo(handle, csr_row_ptr, nnz, m, coo_row_ind, idx_base);

        if(m < 0 || nnz < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || nnz < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // For testing, assemble a COO matrix and convert it to CSR first (on host)

    // Host structures
    std::vector<int> hcoo_row_ind(nnz);
    std::vector<int> hcoo_row_ind_gold(nnz);
    std::vector<int> hcoo_col_ind(nnz);
    std::vector<float> hcoo_val(nnz);

    // Sample initial COO matrix on CPU
    srand(12345ULL);
    gen_matrix_coo(m, n, nnz, hcoo_row_ind_gold, hcoo_col_ind, hcoo_val, idx_base);

    // Convert COO to CSR
    std::vector<int> hcsr_row_ptr(m + 1);

    // csr2coo on host
    for(int i = 0; i < nnz; ++i)
    {
        ++hcsr_row_ptr[hcoo_row_ind_gold[i] + 1 - idx_base];
    }

    hcsr_row_ptr[0] = idx_base;
    for(int i = 0; i < m; ++i)
    {
        hcsr_row_ptr[i + 1] += hcsr_row_ptr[i];
    }

    // Allocate memory on the device
    auto dcsr_row_ptr_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcoo_row_ind_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcoo_row_ind = (int*)dcoo_row_ind_managed.get();

    if(!dcsr_row_ptr || !dcoo_row_ind)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dcsr_row_ptr || !dcoo_row_ind");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(
            hipsparseXcsr2coo(handle, dcsr_row_ptr, nnz, m, dcoo_row_ind, idx_base));

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcoo_row_ind.data(), dcoo_row_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, nnz, hcoo_row_ind_gold.data(), hcoo_row_ind.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            hipsparseXcsr2coo(handle, dcsr_row_ptr, nnz, m, dcoo_row_ind, idx_base);
        }

        double gpu_time_used = get_time_us();

        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            hipsparseXcsr2coo(handle, dcsr_row_ptr, nnz, m, dcoo_row_ind, idx_base);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        double bandwidth = sizeof(int) * (nnz + m + 1) / gpu_time_used / 1e6;

        printf("m\t\tn\t\tnnz\t\tGB/s\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\t%0.2lf\n", m, n, nnz, bandwidth, gpu_time_used);
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSR2COO_HPP
