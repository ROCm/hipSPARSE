/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_ELL2CSR_HPP
#define TESTING_ELL2CSR_HPP

#include "hipsparse_test_unique_ptr.hpp"
#include "hipsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <hipsparse.h>
#include <algorithm>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

#define ELL_IND_ROW(i, el, m, width) (el) * (m) + (i)
#define ELL_IND_EL(i, el, m, width) (el) + (width) * (i)
#define ELL_IND(i, el, m, width) ELL_IND_ROW(i, el, m, width)

template <typename T>
void testing_ell2csr_bad_arg(void)
{
    int m         = 100;
    int n         = 100;
    int ell_width = 100;
    int safe_size = 100;
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t csr_descr = unique_ptr_csr_descr->descr;

    std::unique_ptr<descr_struct> unique_ptr_ell_descr(new descr_struct);
    hipsparseMatDescr_t ell_descr = unique_ptr_ell_descr->descr;

    auto ell_col_ind_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_row_ptr_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};

    int* ell_col_ind = (int*)ell_col_ind_managed.get();
    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();

    if(!ell_col_ind || !csr_row_ptr)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // ELL to CSR conversion is a two step process - test both functions for bad arguments

    // Step 1: Determine number of non-zero elements of CSR storage format
    int csr_nnz;

    // Testing for (ell_col_ind == nullptr)
    {
        int* ell_col_ind_null = nullptr;

        status = hipsparseXell2csrNnz(
            handle, m, n, ell_descr, ell_width, ell_col_ind_null, csr_descr, csr_row_ptr, &csr_nnz);
        verify_hipsparse_status_invalid_pointer(status, "Error: ell_col_ind is nullptr");
    }

    // Testing for (csr_row_ptr == nullptr)
    {
        int* csr_row_ptr_null = nullptr;

        status = hipsparseXell2csrNnz(
            handle, m, n, ell_descr, ell_width, ell_col_ind, csr_descr, csr_row_ptr_null, &csr_nnz);
        verify_hipsparse_status_invalid_pointer(status, "Error: ell_width is nullptr");
    }

    // Testing for (csr_nnz == nullptr)
    {
        int* csr_nnz_null = nullptr;

        status = hipsparseXell2csrNnz(
            handle, m, n, ell_descr, ell_width, ell_col_ind, csr_descr, csr_row_ptr, csr_nnz_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_nnz is nullptr");
    }

    // Testing for (ell_descr == nullptr)
    {
        hipsparseMatDescr_t ell_descr_null = nullptr;

        status = hipsparseXell2csrNnz(
            handle, m, n, ell_descr_null, ell_width, ell_col_ind, csr_descr, csr_row_ptr, &csr_nnz);
        verify_hipsparse_status_invalid_pointer(status, "Error: ell_descr is nullptr");
    }

    // Testing for (csr_descr == nullptr)
    {
        hipsparseMatDescr_t csr_descr_null = nullptr;

        status = hipsparseXell2csrNnz(
            handle, m, n, ell_descr, ell_width, ell_col_ind, csr_descr_null, csr_row_ptr, &csr_nnz);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_descr is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXell2csrNnz(
            handle_null, m, n, ell_descr, ell_width, ell_col_ind, csr_descr, csr_row_ptr, &csr_nnz);
        verify_hipsparse_status_invalid_handle(status);
    }

    // Allocate memory for ELL storage format
    auto ell_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto csr_col_ind_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    T* ell_val       = (T*)ell_val_managed.get();
    int* csr_col_ind = (int*)csr_col_ind_managed.get();
    T* csr_val       = (T*)csr_val_managed.get();

    if(!ell_val || !csr_col_ind || !csr_val)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Step 2: Perform the actual conversion

    // Set ell_width to some valid value, to avoid invalid_size status
    ell_width = 10;

    // Testing for (ell_col_ind == nullptr)
    {
        int* ell_col_ind_null = nullptr;

        status = hipsparseXell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind_null,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: ell_col_ind is nullptr");
    }

    // Testing for (ell_val == nullptr)
    {
        T* ell_val_null = nullptr;

        status = hipsparseXell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val_null,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: ell_val is nullptr");
    }

    // Testing for (csr_row_ptr == nullptr)
    {
        int* csr_row_ptr_null = nullptr;

        status = hipsparseXell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr_null,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    // Testing for (csr_col_ind == nullptr)
    {
        int* csr_col_ind_null = nullptr;

        status = hipsparseXell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind_null);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }

    // Testing for (csr_val == nullptr)
    {
        T* csr_val_null = nullptr;

        status = hipsparseXell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val_null,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");
    }

    // Testing for (ell_descr == nullptr)
    {
        hipsparseMatDescr_t ell_descr_null = nullptr;

        status = hipsparseXell2csr(handle,
                                   m,
                                   n,
                                   ell_descr_null,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: ell_descr is nullptr");
    }

    // Testing for (csr_descr == nullptr)
    {
        hipsparseMatDescr_t csr_descr_null = nullptr;

        status = hipsparseXell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr_null,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_descr is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXell2csr(handle_null,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_hipsparse_status_invalid_handle(status);
    }
}

template <typename T>
hipsparseStatus_t testing_ell2csr(Arguments argus)
{
    int m                         = argus.M;
    int n                         = argus.N;
    int safe_size                 = 100;
    hipsparseIndexBase_t ell_base = argus.idx_base;
    hipsparseIndexBase_t csr_base = argus.idx_base2;
    std::string binfile           = "";
    std::string filename          = "";
    hipsparseStatus_t status;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    if(m == -99 && n == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        m = n = safe_size;
    }

    if(argus.timing == 1)
    {
        filename = argus.filename;
    }

    double scale = 0.02;
    if(m > 1000 || n > 1000)
    {
        scale = 2.0 / std::max(m, n);
    }
    int nnz = m * scale * n;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_ell_descr(new descr_struct);
    hipsparseMatDescr_t ell_descr = unique_ptr_ell_descr->descr;

    // Set ELL matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(ell_descr, ell_base));

    std::unique_ptr<descr_struct> unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t csr_descr = unique_ptr_csr_descr->descr;

    // Set CSR matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(csr_descr, csr_base));

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0)
    {
        auto ell_col_ind_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto ell_val_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto csr_row_ptr_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};

        int* ell_col_ind = (int*)ell_col_ind_managed.get();
        T* ell_val       = (T*)ell_val_managed.get();
        int* csr_row_ptr = (int*)csr_row_ptr_managed.get();

        if(!ell_col_ind || !ell_val || !csr_row_ptr)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!ell_col_ind || !ell_val || !csr_row_ptr");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        int ell_width = safe_size;

        // Step 1 - obtain CSR nnz
        int csr_nnz;
        status = hipsparseXell2csrNnz(
            handle, m, n, ell_descr, ell_width, ell_col_ind, csr_descr, csr_row_ptr, &csr_nnz);

        if(m < 0 || n < 0 || ell_width < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || ell_width < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && n >= 0 && ell_width >= 0");
        }

        // Step 2 - perform actual conversion
        auto csr_col_ind_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto csr_val_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* csr_col_ind = (int*)csr_col_ind_managed.get();
        T* csr_val       = (T*)csr_val_managed.get();

        if(!csr_col_ind || !csr_val)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!csr_col_ind || !csr_val");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        status = hipsparseXell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);

        if(m < 0 || n < 0 || ell_width < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || ell_width < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && n >= 0 && ell_width >= 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // For testing, assemble a CSR matrix

    // Host structures
    std::vector<int> hcsr_row_ptr_gold;
    std::vector<int> hcsr_col_ind_gold;
    std::vector<T> hcsr_val_gold;

    // Sample initial CSR matrix on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(binfile.c_str(),
                           m,
                           n,
                           nnz,
                           hcsr_row_ptr_gold,
                           hcsr_col_ind_gold,
                           hcsr_val_gold,
                           csr_base) != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }
    else if(argus.laplacian)
    {
        m = n = gen_2d_laplacian(
            argus.laplacian, hcsr_row_ptr_gold, hcsr_col_ind_gold, hcsr_val_gold, csr_base);
        nnz = hcsr_row_ptr_gold[m];
    }
    else
    {
        std::vector<int> hcoo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(filename.c_str(),
                               m,
                               n,
                               nnz,
                               hcoo_row_ind,
                               hcsr_col_ind_gold,
                               hcsr_val_gold,
                               csr_base) != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            gen_matrix_coo(m, n, nnz, hcoo_row_ind, hcsr_col_ind_gold, hcsr_val_gold, csr_base);
        }

        // Convert COO to CSR
        hcsr_row_ptr_gold.resize(m + 1, 0);
        for(int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptr_gold[hcoo_row_ind[i] + 1 - csr_base];
        }

        hcsr_row_ptr_gold[0] = csr_base;
        for(int i = 0; i < m; ++i)
        {
            hcsr_row_ptr_gold[i + 1] += hcsr_row_ptr_gold[i];
        }
    }

    int csr_nnz_gold = nnz;

    // Allocate memory on the device
    auto dcsr_row_ptr_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T* dcsr_val       = (T*)dcsr_val_managed.get();

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr_gold.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind_gold.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val, hcsr_val_gold.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Convert CSR matrix to ELL format on GPU
    int ell_width;

    CHECK_HIPSPARSE_ERROR(
        hipsparseXcsr2ellWidth(handle, m, csr_descr, dcsr_row_ptr, ell_descr, &ell_width));

    auto dell_col_ind_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(int) * (ell_width * m)), device_free};
    auto dell_val_managed =
        hipsparse_unique_ptr{device_malloc(sizeof(T) * (ell_width * m)), device_free};

    int* dell_col_ind = (int*)dell_col_ind_managed.get();
    T* dell_val       = (T*)dell_val_managed.get();

    if(!dell_col_ind || !dell_val)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dell_col_ind || !dell_val");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

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

    if(argus.unit_check)
    {
        // Determine csr non-zero entries
        int csr_nnz;

        auto dcsr_row_ptr_conv_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};

        int* dcsr_row_ptr_conv = (int*)dcsr_row_ptr_conv_managed.get();

        if(!dcsr_row_ptr_conv)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED, "!dcsr_row_ptr_conv");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        CHECK_HIPSPARSE_ERROR(hipsparseXell2csrNnz(handle,
                                                   m,
                                                   n,
                                                   ell_descr,
                                                   ell_width,
                                                   dell_col_ind,
                                                   csr_descr,
                                                   dcsr_row_ptr_conv,
                                                   &csr_nnz));

        // Check if CSR nnz does match
        unit_check_general(1, 1, 1, &csr_nnz_gold, &csr_nnz);

        // Allocate CSR column and values arrays
        auto dcsr_col_ind_conv_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(int) * csr_nnz), device_free};
        auto dcsr_val_conv_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(T) * csr_nnz), device_free};

        int* dcsr_col_ind_conv = (int*)dcsr_col_ind_conv_managed.get();
        T* dcsr_val_conv       = (T*)dcsr_val_conv_managed.get();

        if(!dcsr_col_ind_conv || !dcsr_val_conv)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!dcsr_col_ind_conv || !dcsr_val_conv");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        // Perform actual CSR conversion
        CHECK_HIPSPARSE_ERROR(hipsparseXell2csr(handle,
                                                m,
                                                n,
                                                ell_descr,
                                                ell_width,
                                                dell_val,
                                                dell_col_ind,
                                                csr_descr,
                                                dcsr_val_conv,
                                                dcsr_row_ptr_conv,
                                                dcsr_col_ind_conv));

        // Verification host structures
        std::vector<int> hcsr_row_ptr(m + 1);
        std::vector<int> hcsr_col_ind(csr_nnz);
        std::vector<T> hcsr_val(csr_nnz);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr.data(), dcsr_row_ptr_conv, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind.data(), dcsr_col_ind_conv, sizeof(int) * csr_nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val.data(), dcsr_val_conv, sizeof(T) * csr_nnz, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_gold.data(), hcsr_row_ptr.data());
        unit_check_general(1, csr_nnz, 1, hcsr_col_ind_gold.data(), hcsr_col_ind.data());
        unit_check_general(1, csr_nnz, 1, hcsr_val_gold.data(), hcsr_val.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            auto dcsr_row_ptr_conv_managed =
                hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};

            int* dcsr_row_ptr_conv = (int*)dcsr_row_ptr_conv_managed.get();

            int csr_nnz;
            hipsparseXell2csrNnz(handle,
                                 m,
                                 n,
                                 ell_descr,
                                 ell_width,
                                 dell_col_ind,
                                 csr_descr,
                                 dcsr_row_ptr_conv,
                                 &csr_nnz);

            auto dcsr_col_ind_conv_managed =
                hipsparse_unique_ptr{device_malloc(sizeof(int) * csr_nnz), device_free};
            auto dcsr_val_conv_managed =
                hipsparse_unique_ptr{device_malloc(sizeof(T) * csr_nnz), device_free};

            int* dcsr_col_ind_conv = (int*)dcsr_col_ind_conv_managed.get();
            T* dcsr_val_conv       = (T*)dcsr_val_conv_managed.get();

            hipsparseXell2csr(handle,
                              m,
                              n,
                              ell_descr,
                              ell_width,
                              dell_val,
                              dell_col_ind,
                              csr_descr,
                              dcsr_val_conv,
                              dcsr_row_ptr_conv,
                              dcsr_col_ind_conv);
        }

        double gpu_time_used = get_time_us();

        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            auto dcsr_row_ptr_conv_managed =
                hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};

            int* dcsr_row_ptr_conv = (int*)dcsr_row_ptr_conv_managed.get();

            int csr_nnz;
            hipsparseXell2csrNnz(handle,
                                 m,
                                 n,
                                 ell_descr,
                                 ell_width,
                                 dell_col_ind,
                                 csr_descr,
                                 dcsr_row_ptr_conv,
                                 &csr_nnz);

            auto dcsr_col_ind_conv_managed =
                hipsparse_unique_ptr{device_malloc(sizeof(int) * csr_nnz), device_free};
            auto dcsr_val_conv_managed =
                hipsparse_unique_ptr{device_malloc(sizeof(T) * csr_nnz), device_free};

            int* dcsr_col_ind_conv = (int*)dcsr_col_ind_conv_managed.get();
            T* dcsr_val_conv       = (T*)dcsr_val_conv_managed.get();

            hipsparseXell2csr(handle,
                              m,
                              n,
                              ell_descr,
                              ell_width,
                              dell_val,
                              dell_col_ind,
                              csr_descr,
                              dcsr_val_conv,
                              dcsr_row_ptr_conv,
                              dcsr_col_ind_conv);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        printf("m\t\tn\t\tnnz\t\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\n", m, n, nnz, gpu_time_used);
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_ELL2CSR_HPP
