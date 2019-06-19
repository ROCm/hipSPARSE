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
#ifndef TESTING_CSR2HYB_HPP
#define TESTING_CSR2HYB_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

#define ELL_IND_ROW(i, el, m, width) (el) * (m) + (i)
#define ELL_IND_EL(i, el, m, width) (el) + (width) * (i)
#define ELL_IND(i, el, m, width) ELL_IND_ROW(i, el, m, width)

struct test_hyb
{
    int                     m;
    int                     n;
    hipsparseHybPartition_t partition;
    int                     ell_nnz;
    int                     ell_width;
    int*                    ell_col_ind;
    void*                   ell_val;
    int                     coo_nnz;
    int*                    coo_row_ind;
    int*                    coo_col_ind;
    void*                   coo_val;
};

template <typename T>
void testing_csr2hyb_bad_arg(void)
{
    int               m         = 100;
    int               n         = 100;
    int               safe_size = 100;
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<hyb_struct> unique_ptr_hyb(new hyb_struct);
    hipsparseHybMat_t           hyb = unique_ptr_hyb->hyb;

    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* csr_col_ind = (int*)csr_col_ind_managed.get();
    T*   csr_val     = (T*)csr_val_managed.get();

    if(!csr_row_ptr || !csr_col_ind || !csr_val)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing for(csr_row_ptr == nullptr)
    {
        int* csr_row_ptr_null = nullptr;

        status = hipsparseXcsr2hyb(handle,
                                   m,
                                   n,
                                   descr,
                                   csr_val,
                                   csr_row_ptr_null,
                                   csr_col_ind,
                                   hyb,
                                   0,
                                   HIPSPARSE_HYB_PARTITION_AUTO);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }
    // Testing for(csr_col_ind == nullptr)
    {
        int* csr_col_ind_null = nullptr;

        status = hipsparseXcsr2hyb(handle,
                                   m,
                                   n,
                                   descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind_null,
                                   hyb,
                                   0,
                                   HIPSPARSE_HYB_PARTITION_AUTO);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }
    // Testing for(csr_val == nullptr)
    {
        T* csr_val_null = nullptr;

        status = hipsparseXcsr2hyb(handle,
                                   m,
                                   n,
                                   descr,
                                   csr_val_null,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   hyb,
                                   0,
                                   HIPSPARSE_HYB_PARTITION_AUTO);
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");
    }
    // Testing for(handle == nullptr)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXcsr2hyb(handle_null,
                                   m,
                                   n,
                                   descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   hyb,
                                   0,
                                   HIPSPARSE_HYB_PARTITION_AUTO);
        verify_hipsparse_status_invalid_handle(status);
    }
}

template <typename T>
hipsparseStatus_t testing_csr2hyb(Arguments argus)
{
    int                     m              = argus.M;
    int                     n              = argus.N;
    int                     safe_size      = 100;
    hipsparseIndexBase_t    idx_base       = argus.idx_base;
    hipsparseHybPartition_t part           = argus.part;
    int                     user_ell_width = argus.ell_width;
    std::string             binfile        = "";
    std::string             filename       = "";
    hipsparseStatus_t       status;

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
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    std::unique_ptr<hyb_struct> unique_ptr_hyb(new hyb_struct);
    hipsparseHybMat_t           hyb = unique_ptr_hyb->hyb;

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0)
    {
        auto csr_row_ptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto csr_col_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto csr_val_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
        int* csr_col_ind = (int*)csr_col_ind_managed.get();
        T*   csr_val     = (T*)csr_val_managed.get();

        if(!csr_row_ptr || !csr_col_ind || !csr_val)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!csr_row_ptr || !csr_col_ind || !csr_val");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        status = hipsparseXcsr2hyb(
            handle, m, n, descr, csr_val, csr_row_ptr, csr_col_ind, hyb, user_ell_width, part);

        if(m < 0 || n < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: m < 0 || n < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "m >= 0 && n >= 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // For testing, assemble a COO matrix and convert it to CSR first (on host)

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcoo_row_ind;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;

    // Sample initial COO matrix on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(
               binfile.c_str(), m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }
    else if(argus.laplacian)
    {
        m = n = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base);
        nnz   = hcsr_row_ptr[m];
    }
    else
    {
        if(filename != "")
        {
            if(read_mtx_matrix(
                   filename.c_str(), m, n, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, idx_base)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            gen_matrix_coo(m, n, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, idx_base);
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

    // Allocate memory on the device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val     = (T*)dcsr_val_managed.get();

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // User given ELL width check
    if(part == HIPSPARSE_HYB_PARTITION_USER)
    {
        // ELL width -33 means we take a reasonable pre-computed width
        if(user_ell_width == -33)
        {
            user_ell_width = nnz / m;
        }

        // Test invalid user_ell_width
        int max_allowed_ell_nnz_per_row = (2 * nnz - 1) / m + 1;
        if(user_ell_width < 0 || user_ell_width > max_allowed_ell_nnz_per_row)
        {
            status = hipsparseXcsr2hyb(handle,
                                       m,
                                       n,
                                       descr,
                                       dcsr_val,
                                       dcsr_row_ptr,
                                       dcsr_col_ind,
                                       hyb,
                                       user_ell_width,
                                       part);

            verify_hipsparse_status_invalid_value(
                status, "Error: user_ell_width < 0 || user_ell_width > max_ell_width");

            return HIPSPARSE_STATUS_SUCCESS;
        }
    }

    // Max width check
    if(part == HIPSPARSE_HYB_PARTITION_MAX)
    {
        // Compute max ELL width
        int ell_max_width = 0;
        for(int i = 0; i < m; ++i)
        {
            ell_max_width = std::max(hcsr_row_ptr[i + 1] - hcsr_row_ptr[i], ell_max_width);
        }

        int width_limit = (2 * nnz - 1) / m + 1;
        if(ell_max_width > width_limit)
        {
            status = hipsparseXcsr2hyb(handle,
                                       m,
                                       n,
                                       descr,
                                       dcsr_val,
                                       dcsr_row_ptr,
                                       dcsr_col_ind,
                                       hyb,
                                       user_ell_width,
                                       part);

            verify_hipsparse_status_invalid_value(status, "ell_max_width > width_limit");
            return HIPSPARSE_STATUS_SUCCESS;
        }
    }

    // Host structures for verification
    std::vector<int> hhyb_ell_col_ind_gold;
    std::vector<T>   hhyb_ell_val_gold;
    std::vector<int> hhyb_coo_row_ind_gold;
    std::vector<int> hhyb_coo_col_ind_gold;
    std::vector<T>   hhyb_coo_val_gold;

    // Host csr2hyb conversion
    int ell_width = 0;
    int ell_nnz   = 0;
    int coo_nnz   = 0;

    if(part == HIPSPARSE_HYB_PARTITION_AUTO || part == HIPSPARSE_HYB_PARTITION_USER)
    {
        if(part == HIPSPARSE_HYB_PARTITION_AUTO)
        {
            // ELL width is average nnz per row
            ell_width = (nnz - 1) / m + 1;
        }
        else
        {
            // User given ELL width
            ell_width = user_ell_width;
        }

        ell_nnz = ell_width * m;

        // Determine COO nnz
        for(int i = 0; i < m; ++i)
        {
            int row_nnz = hcsr_row_ptr[i + 1] - hcsr_row_ptr[i];
            if(row_nnz > ell_width)
            {
                coo_nnz += row_nnz - ell_width;
            }
        }
    }
    else if(part == HIPSPARSE_HYB_PARTITION_MAX)
    {
        // Determine max nnz per row
        for(int i = 0; i < m; ++i)
        {
            int row_nnz = hcsr_row_ptr[i + 1] - hcsr_row_ptr[i];
            ell_width   = (row_nnz > ell_width) ? row_nnz : ell_width;
        }
        ell_nnz = ell_width * m;
    }

    // Allocate host memory
    // ELL
    hhyb_ell_col_ind_gold.resize(ell_nnz);
    hhyb_ell_val_gold.resize(ell_nnz);
    // COO
    hhyb_coo_row_ind_gold.resize(coo_nnz);
    hhyb_coo_col_ind_gold.resize(coo_nnz);
    hhyb_coo_val_gold.resize(coo_nnz);

    // Fill HYB
    int coo_idx = 0;
    for(int i = 0; i < m; ++i)
    {
        int p = 0;
        for(int j = hcsr_row_ptr[i] - idx_base; j < hcsr_row_ptr[i + 1] - idx_base; ++j)
        {
            if(p < ell_width)
            {
                int idx                    = ELL_IND(i, p++, m, ell_width);
                hhyb_ell_col_ind_gold[idx] = hcsr_col_ind[j];
                hhyb_ell_val_gold[idx]     = hcsr_val[j];
            }
            else
            {
                hhyb_coo_row_ind_gold[coo_idx] = i + idx_base;
                hhyb_coo_col_ind_gold[coo_idx] = hcsr_col_ind[j];
                hhyb_coo_val_gold[coo_idx]     = hcsr_val[j];
                ++coo_idx;
            }
        }
        for(int j = hcsr_row_ptr[i + 1] - hcsr_row_ptr[i]; j < ell_width; ++j)
        {
            int idx                    = ELL_IND(i, p++, m, ell_width);
            hhyb_ell_col_ind_gold[idx] = -1;
            hhyb_ell_val_gold[idx]     = static_cast<T>(0);
        }
    }

    // Allocate verification structures
    std::vector<int> hhyb_ell_col_ind(ell_nnz);
    std::vector<T>   hhyb_ell_val(ell_nnz);
    std::vector<int> hhyb_coo_row_ind(coo_nnz);
    std::vector<int> hhyb_coo_col_ind(coo_nnz);
    std::vector<T>   hhyb_coo_val(coo_nnz);

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseXcsr2hyb(
            handle, m, n, descr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, hyb, user_ell_width, part));

        // Copy output from device to host
        test_hyb* dhyb = (test_hyb*)hyb;

        // Check if sizes match
        unit_check_general(1, 1, 1, &m, &dhyb->m);
        unit_check_general(1, 1, 1, &n, &dhyb->n);
        unit_check_general(1, 1, 1, &ell_width, &dhyb->ell_width);
        unit_check_general(1, 1, 1, &ell_nnz, &dhyb->ell_nnz);
        unit_check_general(1, 1, 1, &coo_nnz, &dhyb->coo_nnz);

        CHECK_HIP_ERROR(hipMemcpy(hhyb_ell_col_ind.data(),
                                  dhyb->ell_col_ind,
                                  sizeof(int) * ell_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hhyb_ell_val.data(), dhyb->ell_val, sizeof(T) * ell_nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hhyb_coo_row_ind.data(),
                                  dhyb->coo_row_ind,
                                  sizeof(int) * coo_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hhyb_coo_col_ind.data(),
                                  dhyb->coo_col_ind,
                                  sizeof(int) * coo_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hhyb_coo_val.data(), dhyb->coo_val, sizeof(T) * coo_nnz, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, ell_nnz, 1, hhyb_ell_col_ind_gold.data(), hhyb_ell_col_ind.data());
        unit_check_general(1, ell_nnz, 1, hhyb_ell_val_gold.data(), hhyb_ell_val.data());
        unit_check_general(1, coo_nnz, 1, hhyb_coo_row_ind_gold.data(), hhyb_coo_row_ind.data());
        unit_check_general(1, coo_nnz, 1, hhyb_coo_col_ind_gold.data(), hhyb_coo_col_ind.data());
        unit_check_general(1, coo_nnz, 1, hhyb_coo_val_gold.data(), hhyb_coo_val.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            hipsparseXcsr2hyb(handle,
                              m,
                              n,
                              descr,
                              dcsr_val,
                              dcsr_row_ptr,
                              dcsr_col_ind,
                              hyb,
                              user_ell_width,
                              part);
        }

        double gpu_time_used = get_time_us();

        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            hipsparseXcsr2hyb(handle,
                              m,
                              n,
                              descr,
                              dcsr_val,
                              dcsr_row_ptr,
                              dcsr_col_ind,
                              hyb,
                              user_ell_width,
                              part);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        printf("m\t\tn\t\tnnz\t\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\n", m, n, nnz, gpu_time_used);
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSR2HYB_HPP
