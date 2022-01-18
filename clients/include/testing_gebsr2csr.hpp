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
#ifndef TESTING_GEBSR2CSR_HPP
#define TESTING_GEBSR2CSR_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse.h>
#include <iostream>
#include <string>
using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_gebsr2csr_bad_arg(void)
{

#ifdef __HIP_PLATFORM_NVIDIA__
    // do not test for bad args
    return;
#endif
    int                  m             = 100;
    int                  n             = 100;
    int                  safe_size     = 100;
    int                  row_block_dim = 2;
    int                  col_block_dim = 2;
    hipsparseIndexBase_t csr_idx_base  = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t bsr_idx_base  = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseDirection_t dir           = HIPSPARSE_DIRECTION_ROW;
    hipsparseStatus_t    status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t            csr_descr = unique_ptr_csr_descr->descr;
    std::unique_ptr<descr_struct>  unique_ptr_bsr_descr(new descr_struct);
    hipsparseMatDescr_t            bsr_descr = unique_ptr_bsr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, csr_idx_base);
    hipsparseSetMatIndexBase(bsr_descr, bsr_idx_base);

    auto bsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};

    auto bsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* bsr_row_ptr = (int*)bsr_row_ptr_managed.get();
    int* bsr_col_ind = (int*)bsr_col_ind_managed.get();
    T*   bsr_val     = (T*)bsr_val_managed.get();
    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* csr_col_ind = (int*)csr_col_ind_managed.get();
    T*   csr_val     = (T*)csr_val_managed.get();

    if(!bsr_row_ptr || !bsr_col_ind || !bsr_val || !csr_row_ptr || !csr_col_ind || !csr_val)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing hipsparseXgebsr2csr()

    // Test invalid handle
    status = hipsparseXgebsr2csr(nullptr,
                                 dir,
                                 m,
                                 n,
                                 bsr_descr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 row_block_dim,
                                 col_block_dim,
                                 csr_descr,
                                 csr_val,
                                 csr_row_ptr,
                                 csr_col_ind);
    verify_hipsparse_status_invalid_handle(status);

    // Test invalid pointers
    status = hipsparseXgebsr2csr(handle,
                                 dir,
                                 m,
                                 n,
                                 nullptr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 row_block_dim,
                                 col_block_dim,
                                 csr_descr,
                                 csr_val,
                                 csr_row_ptr,
                                 csr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_descr is nullptr");

    status = hipsparseXgebsr2csr(handle,
                                 dir,
                                 m,
                                 n,
                                 bsr_descr,
                                 (T*)nullptr,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 row_block_dim,
                                 col_block_dim,
                                 csr_descr,
                                 csr_val,
                                 csr_row_ptr,
                                 csr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_val is nullptr");

    status = hipsparseXgebsr2csr(handle,
                                 dir,
                                 m,
                                 n,
                                 bsr_descr,
                                 bsr_val,
                                 nullptr,
                                 bsr_col_ind,
                                 row_block_dim,
                                 col_block_dim,
                                 csr_descr,
                                 csr_val,
                                 csr_row_ptr,
                                 csr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr is nullptr");

    status = hipsparseXgebsr2csr(handle,
                                 dir,
                                 m,
                                 n,
                                 bsr_descr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 nullptr,
                                 row_block_dim,
                                 col_block_dim,
                                 csr_descr,
                                 csr_val,
                                 csr_row_ptr,
                                 csr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind is nullptr");

    status = hipsparseXgebsr2csr(handle,
                                 dir,
                                 m,
                                 n,
                                 bsr_descr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 row_block_dim,
                                 col_block_dim,
                                 nullptr,
                                 csr_val,
                                 csr_row_ptr,
                                 csr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_descr is nullptr");

    status = hipsparseXgebsr2csr(handle,
                                 dir,
                                 m,
                                 n,
                                 bsr_descr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 row_block_dim,
                                 col_block_dim,
                                 csr_descr,
                                 (T*)nullptr,
                                 csr_row_ptr,
                                 csr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");

    status = hipsparseXgebsr2csr(handle,
                                 dir,
                                 m,
                                 n,
                                 bsr_descr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 row_block_dim,
                                 col_block_dim,
                                 csr_descr,
                                 csr_val,
                                 nullptr,
                                 csr_col_ind);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");

    status = hipsparseXgebsr2csr(handle,
                                 dir,
                                 m,
                                 n,
                                 bsr_descr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 row_block_dim,
                                 col_block_dim,
                                 csr_descr,
                                 csr_val,
                                 csr_row_ptr,
                                 nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");

    // Test invalid sizes
    status = hipsparseXgebsr2csr(handle,
                                 dir,
                                 -1,
                                 n,
                                 bsr_descr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 row_block_dim,
                                 col_block_dim,
                                 csr_descr,
                                 csr_val,
                                 csr_row_ptr,
                                 csr_col_ind);
    verify_hipsparse_status_invalid_size(status, "Error: m is invalid");

    status = hipsparseXgebsr2csr(handle,
                                 dir,
                                 m,
                                 -1,
                                 bsr_descr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 row_block_dim,
                                 col_block_dim,
                                 csr_descr,
                                 csr_val,
                                 csr_row_ptr,
                                 csr_col_ind);
    verify_hipsparse_status_invalid_size(status, "Error: n is invalid");

    status = hipsparseXgebsr2csr(handle,
                                 dir,
                                 m,
                                 n,
                                 bsr_descr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 -1,
                                 col_block_dim,
                                 csr_descr,
                                 csr_val,
                                 csr_row_ptr,
                                 csr_col_ind);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim is invalid");

    status = hipsparseXgebsr2csr(handle,
                                 dir,
                                 m,
                                 n,
                                 bsr_descr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 row_block_dim,
                                 -1,
                                 csr_descr,
                                 csr_val,
                                 csr_row_ptr,
                                 csr_col_ind);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim is invalid");
}

template <typename T>
hipsparseStatus_t testing_gebsr2csr(Arguments argus)
{

    int                  m             = argus.M;
    int                  n             = argus.N;
    int                  row_block_dim = argus.row_block_dimA;
    int                  col_block_dim = argus.col_block_dimA;
    hipsparseIndexBase_t csr_idx_base  = argus.idx_base;
    hipsparseIndexBase_t bsr_idx_base  = argus.idx_base2;
    hipsparseDirection_t dir           = argus.dirA;
    std::string          binfile       = "";
    std::string          filename      = "";
    hipsparseStatus_t    status;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    int safe_size = 100;
    if(m == -99 && n == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        m = n = safe_size;
    }

    if(argus.timing == 1)
    {
        filename = argus.filename;
    }

    int mb = -1;
    int nb = -1;
    if(row_block_dim > 0 && col_block_dim > 0)
    {
        mb = m * row_block_dim;
        nb = n * col_block_dim;
    }

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t            csr_descr = unique_ptr_csr_descr->descr;
    std::unique_ptr<descr_struct>  unique_ptr_bsr_descr(new descr_struct);
    hipsparseMatDescr_t            bsr_descr = unique_ptr_bsr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, csr_idx_base);
    hipsparseSetMatIndexBase(bsr_descr, bsr_idx_base);

    // Argument sanity check before allocating invalid memory
    if(mb <= 0 || nb <= 0 || row_block_dim <= 0 || col_block_dim <= 0)
    {
        auto dbsr_row_ptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dbsr_col_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dbsr_val_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dcsr_row_ptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dcsr_col_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dcsr_val_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();
        int* dbsr_col_ind = (int*)dbsr_col_ind_managed.get();
        T*   dbsr_val     = (T*)dbsr_val_managed.get();
        int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
        int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
        T*   dcsr_val     = (T*)dcsr_val_managed.get();

        if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dcsr_row_ptr || !dcsr_col_ind
           || !dcsr_val)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || "
                                            "!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        status = hipsparseXgebsr2csr(handle,
                                     dir,
                                     mb,
                                     nb,
                                     bsr_descr,
                                     dbsr_val,
                                     dbsr_row_ptr,
                                     dbsr_col_ind,
                                     row_block_dim,
                                     col_block_dim,
                                     csr_descr,
                                     dcsr_val,
                                     dcsr_row_ptr,
                                     dcsr_col_ind);

        if(mb < 0 || nb < 0 || row_block_dim < 0 || col_block_dim < 0)
        {
            verify_hipsparse_status_invalid_size(
                status, "Error: mb < 0 || nb < 0 || row_block_dim < 0 || col_block_dim < 0");
        }
        else
        {
            verify_hipsparse_status_success(
                status, "mb >= 0 && nb >= 0 && row_block_dim >= 0 && col_block_dim >= 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Read or construct CSR matrix
    std::vector<int> bsr_row_ptr;
    std::vector<int> bsr_col_ind;
    std::vector<T>   bsr_val;
    int              nnzb;
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(
               binfile.c_str(), mb, nb, nnzb, bsr_row_ptr, bsr_col_ind, bsr_val, bsr_idx_base)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }
    else if(argus.laplacian)
    {
        mb = nb
            = gen_2d_laplacian(argus.laplacian, bsr_row_ptr, bsr_col_ind, bsr_val, bsr_idx_base);
        nnzb = bsr_row_ptr[m];
    }
    else
    {
        std::vector<int> coo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(
                   filename.c_str(), mb, nb, nnzb, coo_row_ind, bsr_col_ind, bsr_val, bsr_idx_base)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            double scale = 0.02;
            if(m > 1000 || n > 1000)
            {
                scale = 2.0 / std::max(mb, nb);
            }
            nnzb = mb * scale * nb;
            gen_matrix_coo(mb, nb, nnzb, coo_row_ind, bsr_col_ind, bsr_val, bsr_idx_base);
        }

        // Convert COO to CSR
        bsr_row_ptr.resize(mb + 1, 0);
        for(int i = 0; i < nnzb; ++i)
        {
            ++bsr_row_ptr[coo_row_ind[i] + 1 - bsr_idx_base];
        }

        bsr_row_ptr[0] = bsr_idx_base;
        for(int i = 0; i < mb; ++i)
        {
            bsr_row_ptr[i + 1] += bsr_row_ptr[i];
        }
    }

    m       = mb * row_block_dim;
    n       = nb * col_block_dim;
    int nnz = nnzb * row_block_dim * col_block_dim;
    // Now use the csr matrix as the symbolic for the gebsr matrix.
    bsr_val.resize(nnz);
    int idx = 0;
    switch(dir)
    {
    case HIPSPARSE_DIRECTION_COLUMN:
    {
        for(int i = 0; i < mb; ++i)
        {
            for(int r = 0; r < row_block_dim; ++r)
            {
                for(int k = bsr_row_ptr[i] - bsr_idx_base; k < bsr_row_ptr[i + 1] - bsr_idx_base;
                    ++k)
                {
                    int j = bsr_col_ind[k] - bsr_idx_base;
                    for(int c = 0; c < col_block_dim; ++c)
                    {
                        bsr_val[k * row_block_dim * col_block_dim + c * row_block_dim + r]
                            = make_DataType<T>(++idx);
                    }
                }
            }
        }
        break;
    }

    case HIPSPARSE_DIRECTION_ROW:
    {
        for(int i = 0; i < mb; ++i)
        {
            for(int r = 0; r < row_block_dim; ++r)
            {
                for(int k = bsr_row_ptr[i] - bsr_idx_base; k < bsr_row_ptr[i + 1] - bsr_idx_base;
                    ++k)
                {
                    int j = bsr_col_ind[k] - bsr_idx_base;
                    for(int c = 0; c < col_block_dim; ++c)
                    {
                        bsr_val[k * row_block_dim * col_block_dim + r * col_block_dim + c]
                            = make_DataType<T>(++idx);
                    }
                }
            }
        }
        break;
    }
    }

    // Allocate memory on the device
    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};
    auto dbsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnzb), device_free};
    auto dbsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();
    int* dbsr_col_ind = (int*)dbsr_col_ind_managed.get();
    T*   dbsr_val     = (T*)dbsr_val_managed.get();
    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val     = (T*)dcsr_val_managed.get();

    if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || "
                                        "!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_row_ptr, bsr_row_ptr.data(), sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_col_ind, bsr_col_ind.data(), sizeof(int) * nnzb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_val, bsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        // DEVICE
        CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2csr(handle,
                                                  dir,
                                                  mb,
                                                  nb,
                                                  bsr_descr,
                                                  dbsr_val,
                                                  dbsr_row_ptr,
                                                  dbsr_col_ind,
                                                  row_block_dim,
                                                  col_block_dim,
                                                  csr_descr,
                                                  dcsr_val,
                                                  dcsr_row_ptr,
                                                  dcsr_col_ind));

        // Host CSR matrix
        std::vector<int> dh_csr_row_ptr(m + 1);
        std::vector<int> dh_csr_col_ind(nnz);
        std::vector<T>   dh_csr_val(nnz);

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(
            dh_csr_row_ptr.data(), dcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            dh_csr_col_ind.data(), dcsr_col_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(dh_csr_val.data(), dcsr_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        // Host CSR matrix
        std::vector<int> h_csr_row_ptr(m + 1);
        std::vector<int> h_csr_col_ind(nnz);
        std::vector<T>   h_csr_val(nnz);

        // Host gebsr2csr
        host_gebsr_to_csr<T>(dir,
                             mb,
                             nb,
                             nnzb,
                             bsr_val,
                             bsr_row_ptr,
                             bsr_col_ind,
                             row_block_dim,
                             col_block_dim,
                             bsr_idx_base,
                             h_csr_val,
                             h_csr_row_ptr,
                             h_csr_col_ind,
                             csr_idx_base);

        // Unit check
        unit_check_general(1, m + 1, 1, dh_csr_row_ptr.data(), h_csr_row_ptr.data());
        unit_check_general(1, nnz, 1, dh_csr_col_ind.data(), h_csr_col_ind.data());
        unit_check_general(1, nnz, 1, dh_csr_val.data(), h_csr_val.data());
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_GEBSR2CSR_HPP
