/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_CSR2ELL_HPP
#define TESTING_CSR2ELL_HPP

#include "hipsparse_test_unique_ptr.hpp"
#include "hipsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <hipsparse.h>
#include <algorithm>

using namespace hipsparse;
using namespace hipsparse_test;

#define ELL_IND_ROW(i, el, m, width) (el) * (m) + (i)
#define ELL_IND_EL(i, el, m, width) (el) + (width) * (i)
#define ELL_IND(i, el, m, width) ELL_IND_ROW(i, el, m, width)

template <typename T>
void testing_csr2ell_bad_arg(void)
{
    int m         = 100;
    int safe_size = 100;
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t csr_descr = unique_ptr_csr_descr->descr;

    std::unique_ptr<descr_struct> unique_ptr_ell_descr(new descr_struct);
    hipsparseMatDescr_t ell_descr = unique_ptr_ell_descr->descr;

    auto csr_row_ptr_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_col_ind_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* csr_col_ind = (int*)csr_col_ind_managed.get();
    T* csr_val                 = (T*)csr_val_managed.get();

    if(!csr_row_ptr || !csr_col_ind || !csr_val)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // CSR to ELL conversion is a two step process - test both functions for bad arguments

    // Step 1: Determine number of non-zero elements of ELL storage format
    int ell_width;

    // Testing for (csr_row_ptr == nullptr)
    {
        int* csr_row_ptr_null = nullptr;

        status =
            hipsparseXcsr2ellWidth(handle, m, csr_descr, csr_row_ptr_null, ell_descr, &ell_width);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    // Testing for (ell_widht == nullptr)
    {
        int* ell_width_null = nullptr;

        status =
            hipsparseXcsr2ellWidth(handle, m, csr_descr, csr_row_ptr, ell_descr, ell_width_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: ell_width is nullptr");
    }

    // Testing for (csr_descr == nullptr)
    {
        hipsparseMatDescr_t csr_descr_null = nullptr;

        status =
            hipsparseXcsr2ellWidth(handle, m, csr_descr_null, csr_row_ptr, ell_descr, &ell_width);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_descr is nullptr");
    }

    // Testing for (ell_descr == nullptr)
    {
        hipsparseMatDescr_t ell_descr_null = nullptr;

        status =
            hipsparseXcsr2ellWidth(handle, m, csr_descr, csr_row_ptr, ell_descr_null, &ell_width);
        verify_hipsparse_status_invalid_pointer(status, "Error: ell_descr is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        hipsparseHandle_t handle_null = nullptr;

        status =
            hipsparseXcsr2ellWidth(handle_null, m, csr_descr, csr_row_ptr, ell_descr, &ell_width);
        verify_hipsparse_status_invalid_handle(status);
    }

    // Allocate memory for ELL storage format
    auto ell_col_ind_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto ell_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* ell_col_ind = (int*)ell_col_ind_managed.get();
    T* ell_val                 = (T*)ell_val_managed.get();

    if(!ell_col_ind || !ell_val)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Step 2: Perform the actual conversion

    // Set ell_width to some valid value, to avoid invalid_size status
    ell_width = 10;

    // Testing for (csr_row_ptr == nullptr)
    {
        int* csr_row_ptr_null = nullptr;

        status = hipsparseXcsr2ell(handle,
                                   m,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr_null,
                                   csr_col_ind,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    // Testing for (csr_col_ind == nullptr)
    {
        int* csr_col_ind_null = nullptr;

        status = hipsparseXcsr2ell(handle,
                                   m,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind_null,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }

    // Testing for (csr_val == nullptr)
    {
        T* csr_val_null = nullptr;

        status = hipsparseXcsr2ell(handle,
                                   m,
                                   csr_descr,
                                   csr_val_null,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");
    }

    // Testing for (ell_col_ind == nullptr)
    {
        int* ell_col_ind_null = nullptr;

        status = hipsparseXcsr2ell(handle,
                                   m,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: ell_col_ind is nullptr");
    }

    // Testing for (ell_val == nullptr)
    {
        T* ell_val_null = nullptr;

        status = hipsparseXcsr2ell(handle,
                                   m,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   ell_descr,
                                   ell_width,
                                   ell_val_null,
                                   ell_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: ell_val is nullptr");
    }

    // Testing for (csr_descr == nullptr)
    {
        hipsparseMatDescr_t csr_descr_null = nullptr;

        status = hipsparseXcsr2ell(handle,
                                   m,
                                   csr_descr_null,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_descr is nullptr");
    }

    // Testing for (ell_descr == nullptr)
    {
        hipsparseMatDescr_t ell_descr_null = nullptr;

        status = hipsparseXcsr2ell(handle,
                                   m,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   ell_descr_null,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: ell_descr is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsr2ell(handle_null,
                                   m,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind);
        verify_hipsparse_status_invalid_handle(status);
    }
}

template <typename T>
hipsparseStatus_t testing_csr2ell(Arguments argus)
{
    int m               = argus.M;
    int n               = argus.N;
    int safe_size       = 100;
    hipsparseIndexBase_t csr_base = argus.idx_base;
    hipsparseIndexBase_t ell_base = argus.idx_base2;
    hipsparseStatus_t status;

    double scale = 0.02;
    if(m > 1000 || n > 1000)
    {
        scale = 2.0 / std::max(m, n);
    }
    int nnz = m * scale * n;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t csr_descr = unique_ptr_csr_descr->descr;

    // Set CSR matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(csr_descr, csr_base));

    std::unique_ptr<descr_struct> unique_ptr_ell_descr(new descr_struct);
    hipsparseMatDescr_t ell_descr = unique_ptr_ell_descr->descr;

    // Set ELL matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(ell_descr, ell_base));

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0)
    {
        auto csr_row_ptr_managed =
            (m > 0)
                ? hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free}
                : hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size),
                                       device_free};
        auto csr_col_ind_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto csr_val_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
        int* csr_col_ind = (int*)csr_col_ind_managed.get();
        T* csr_val                 = (T*)csr_val_managed.get();

        if(!csr_row_ptr || !csr_col_ind || !csr_val)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!csr_row_ptr || !csr_col_ind || !csr_val");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        // To obtain valid input, csr_row_ptr need to be 0 (because either m, n or nnz is 0)
        hipMemset(csr_row_ptr, 0, sizeof(int) * ((m > 0) ? (m + 1) : safe_size));

        // Step 1
        int ell_width;
        status = hipsparseXcsr2ellWidth(handle, m, csr_descr, csr_row_ptr, ell_descr, &ell_width);

        if(m < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0");
        }

        auto ell_col_ind_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto ell_val_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* ell_col_ind = (int*)ell_col_ind_managed.get();
        T* ell_val                 = (T*)ell_val_managed.get();

        if(!ell_col_ind || !ell_val)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!ell_col_ind || !ell_val");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        // Step 2
        status = hipsparseXcsr2ell(handle,
                                   m,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   ell_descr,
                                   0,
                                   ell_val,
                                   ell_col_ind);

        if(m < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // For testing, assemble a COO matrix and convert it to CSR first (on host)

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcoo_row_ind;
    std::vector<int> hcsr_col_ind;
    std::vector<T> hcsr_val;

    // Sample initial COO matrix on CPU
    srand(12345ULL);
    if(argus.laplacian)
    {
        m = n = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcsr_col_ind, hcsr_val, csr_base);
        nnz   = hcsr_row_ptr[m];
    }
    else
    {
        if(argus.filename != "")
        {
            if(read_mtx_matrix(argus.filename.c_str(),
                               m,
                               n,
                               nnz,
                               hcoo_row_ind,
                               hcsr_col_ind,
                               hcsr_val,
                               csr_base) != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", argus.filename.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            gen_matrix_coo(m, n, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, csr_base);
        }

        // Convert COO to CSR
        hcsr_row_ptr.resize(m + 1, 0);
        for(int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptr[hcoo_row_ind[i] + 1 - csr_base];
        }

        hcsr_row_ptr[0] = csr_base;
        for(int i = 0; i < m; ++i)
        {
            hcsr_row_ptr[i + 1] += hcsr_row_ptr[i];
        }
    }

    // Allocate memory on the device
    auto dcsr_row_ptr_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T* dcsr_val                 = (T*)dcsr_val_managed.get();

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Host csr2ell conversion
    int ell_width_gold = 0;

    // Determine max nnz per row
    for(int i = 0; i < m; ++i)
    {
        int row_nnz = hcsr_row_ptr[i + 1] - hcsr_row_ptr[i];
        ell_width_gold        = (row_nnz > ell_width_gold) ? row_nnz : ell_width_gold;
    }

    int ell_nnz_gold = ell_width_gold * m;

    // Allocate host memory
    std::vector<int> hell_col_ind_gold(ell_nnz_gold);
    std::vector<T> hell_val_gold(ell_nnz_gold);

    // Fill ELL structures
    for(int i = 0; i < m; ++i)
    {
        int p = 0;
        for(int j = hcsr_row_ptr[i] - csr_base; j < hcsr_row_ptr[i + 1] - csr_base; ++j)
        {
            if(p >= ell_width_gold)
            {
                break;
            }

            int idx      = ELL_IND(i, p++, m, ell_width_gold);
            hell_col_ind_gold[idx] = hcsr_col_ind[j] - csr_base + ell_base;
            hell_val_gold[idx]     = hcsr_val[j];
        }
        for(int j = hcsr_row_ptr[i + 1] - hcsr_row_ptr[i]; j < ell_width_gold; ++j)
        {
            int idx      = ELL_IND(i, p++, m, ell_width_gold);
            hell_col_ind_gold[idx] = -1;
            hell_val_gold[idx]     = static_cast<T>(0);
        }
    }

    // Allocate verification structures
    std::vector<int> hell_col_ind(ell_nnz_gold);
    std::vector<T> hell_val(ell_nnz_gold);
    int ell_width;

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(
            hipsparseXcsr2ellWidth(handle, m, csr_descr, dcsr_row_ptr, ell_descr, &ell_width));

        int ell_nnz = ell_width * m;

        // Check if ELL width does match
        unit_check_general(1, 1, 1, &ell_width_gold, &ell_width);
        unit_check_general(1, 1, 1, &ell_nnz_gold, &ell_nnz);

        // Allocate ELL device memory
        auto dell_col_ind_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(int) * ell_nnz), device_free};
        auto dell_val_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(T) * ell_nnz), device_free};

        int* dell_col_ind = (int*)dell_col_ind_managed.get();
        T* dell_val                 = (T*)dell_val_managed.get();

        // Perform actual ELL conversion
        CHECK_HIPSPARSE_ERROR(hipsparseXcsr2ell(handle,
                                                m,
                                                csr_descr,
                                                dcsr_val,
                                                dcsr_row_ptr,
                                                dcsr_col_ind,
                                                ell_descr,
                                                ell_width,
                                                dell_val,
                                                dell_col_ind));

        CHECK_HIP_ERROR(hipMemcpy(hell_col_ind.data(),
                                  dell_col_ind,
                                  sizeof(int) * ell_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hell_val.data(), dell_val, sizeof(T) * ell_nnz, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, ell_nnz, 1, hell_col_ind_gold.data(), hell_col_ind.data());
        unit_check_general(1, ell_nnz, 1, hell_val_gold.data(), hell_val.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            hipsparseXcsr2ellWidth(handle, m, csr_descr, dcsr_row_ptr, ell_descr, &ell_width);
            int ell_nnz = ell_width * m;

            auto dell_col_ind_managed =
                hipsparse_unique_ptr{device_malloc(sizeof(int) * ell_nnz), device_free};
            auto dell_val_managed =
                hipsparse_unique_ptr{device_malloc(sizeof(T) * ell_nnz), device_free};

            int* dell_col_ind = (int*)dell_col_ind_managed.get();
            T* dell_val                 = (T*)dell_val_managed.get();

            hipsparseXcsr2ell(handle,
                              m,
                              csr_descr,
                              dcsr_val,
                              dcsr_row_ptr,
                              dcsr_col_ind,
                              ell_descr,
                              ell_width,
                              dell_val,
                              dell_col_ind);
        }

        double gpu_time_used = get_time_us();

        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            hipsparseXcsr2ellWidth(handle, m, csr_descr, dcsr_row_ptr, ell_descr, &ell_width);
            int ell_nnz = ell_width * m;

            auto dell_col_ind_managed =
                hipsparse_unique_ptr{device_malloc(sizeof(int) * ell_nnz), device_free};
            auto dell_val_managed =
                hipsparse_unique_ptr{device_malloc(sizeof(T) * ell_nnz), device_free};

            int* dell_col_ind = (int*)dell_col_ind_managed.get();
            T* dell_val                 = (T*)dell_val_managed.get();

            hipsparseXcsr2ell(handle,
                              m,
                              csr_descr,
                              dcsr_val,
                              dcsr_row_ptr,
                              dcsr_col_ind,
                              ell_descr,
                              ell_width,
                              dell_val,
                              dell_col_ind);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        printf("m\t\tn\t\tnnz\t\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\n", m, n, nnz, gpu_time_used);
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSR2ELL_HPP
