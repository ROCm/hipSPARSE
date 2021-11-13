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
#ifndef TESTING_GEBSR2GEBSC_HPP
#define TESTING_GEBSR2GEBSC_HPP

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
void testing_gebsr2gebsc_bad_arg(void)
{

#ifdef __HIP_PLATFORM_NVCC__
    // do not test for bad args
    return;
#endif

    hipsparseStatus_t              status;
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    static const size_t safe_size = 100;

    auto bsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* bsr_row_ptr = (int*)bsr_row_ptr_managed.get();
    int* bsr_col_ind = (int*)bsr_col_ind_managed.get();
    T*   bsr_val     = (T*)bsr_val_managed.get();

    auto bsc_row_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsc_col_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsc_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    auto  buffer_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    void* buffer         = buffer_managed.get();

    int* bsc_row_ind = (int*)bsc_row_ind_managed.get();
    int* bsc_col_ptr = (int*)bsc_col_ptr_managed.get();
    T*   bsc_val     = (T*)bsc_val_managed.get();

    if(!bsr_row_ptr || !bsr_col_ind || !bsr_val || !bsc_row_ind || !bsc_col_ptr || !bsc_val)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    size_t buffer_size;
    status = hipsparseXgebsr2gebsc_bufferSize<T>(nullptr,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_handle(status);

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 -1,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: mb is invalid");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 -1,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: nb is invalid");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 -1,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: nnzb is invalid");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_val is nullptr");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 nullptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr is nullptr");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 nullptr,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind is nullptr");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 -1,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim is invalid");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 -1,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim is invalid");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: buffer_size is nullptr");

    // Test hipsparseXgebsr2gebsc()
    status = hipsparseXgebsr2gebsc<T>(nullptr,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_handle(status);

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      -1,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_size(status, "Error: mb is invalid");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      -1,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nb is invalid");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      -1,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nnzb is invalid");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      nullptr,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_size(status, "Error: bsr_val is nullptr");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      nullptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr is nullptr");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      nullptr,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind is nullptr");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      -1,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim is invalid");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      -1,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim is invalid");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      nullptr,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsc_val is nullptr");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      nullptr,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsc_row_ind is nullptr");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      nullptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsc_col_ptr is nullptr");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: buffer is invalid");
}

#define DEVICE_ALLOC(TYPE, NAME, SIZE)                                                         \
    auto  NAME##_managed = hipsparse_unique_ptr{device_malloc(sizeof(TYPE) * SIZE), device_free}; \
    TYPE* NAME           = (TYPE*)NAME##_managed.get()

template <typename T>
hipsparseStatus_t testing_gebsr2gebsc(Arguments argus)
{
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    hipsparseAction_t    action = argus.action;
    hipsparseIndexBase_t base   = argus.idx_base;

    // Argument sanity check before allocating invalid memory
    if(argus.M <= 0 || argus.N <= 0 || argus.row_block_dimA <= 0 || argus.col_block_dimA <= 0)
    {
#if(defined(CUDART_VERSION))
	if(argus.row_block_dimA == 0 || argus.col_block_dimA == 0)
	{
	    return HIPSPARSE_STATUS_SUCCESS;
	}
#endif

        int M             = argus.M;
        int N             = argus.N;
        int row_block_dim = argus.row_block_dimA;
        int col_block_dim = argus.col_block_dimA;

        static const size_t safe_size = 100;

        // Allocate memory on device
        DEVICE_ALLOC(T, dbuffer, safe_size);

        if(!dbuffer)
        {
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        size_t buffer_size;

        status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                     M,
                                                     N,
                                                     safe_size,
                                                     (const T*)nullptr,
                                                     nullptr,
                                                     nullptr,
                                                     row_block_dim,
                                                     col_block_dim,
                                                     &buffer_size);

        if(M < 0 || N < 0 || row_block_dim < 0 || col_block_dim < 0)
        {
            verify_hipsparse_status_invalid_size(
                status, "Error: M < 0 || N < 0 || row_block_dim < 0 || col_block_dim < 0");
        }
        else
        {
            verify_hipsparse_status_success(
                status, "NOT(M < 0 || N < 0 || row_block_dim < 0 || col_block_dim < 0)");
        }

        status = hipsparseXgebsr2gebsc<T>(handle,
                                          M,
                                          N,
                                          safe_size,
                                          (const T*)nullptr,
                                          nullptr,
                                          nullptr,
                                          row_block_dim,
                                          col_block_dim,
                                          (T*)nullptr,
                                          nullptr,
                                          nullptr,
                                          action,
                                          base,
                                          dbuffer);

        if(M < 0 || N < 0 || row_block_dim < 0 || col_block_dim < 0)
        {
            verify_hipsparse_status_invalid_size(
                status, "Error: M < 0 || N < 0 || row_block_dim < 0 || col_block_dim < 0");
        }
        else
        {
            verify_hipsparse_status_success(
                status, "NOT(M < 0 || N < 0 || row_block_dim < 0 || col_block_dim < 0)");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    //
    // Build the gebsr matrix.
    //
    hipsparseDirection_t bsr_dirb          = argus.dirA;
    int                  bsr_mb            = -1;
    int                  bsr_nb            = -1;
    int                  bsr_nnzb          = -1;
    int                  bsr_row_block_dim = -1;
    int                  bsr_col_block_dim = -1;
    std::vector<int>     hbsr_row_ptr;
    std::vector<int>     hbsr_col_ind;
    std::vector<T>       hbsr_val;

    {
        bsr_dirb          = argus.dirA;
        int m             = argus.M;
        int n             = argus.N;
        bsr_row_block_dim = argus.row_block_dimA;
        bsr_col_block_dim = argus.col_block_dimA;

        std::string binfile  = "";
        std::string filename = "";

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

        // Read or construct CSR matrix
        std::vector<int> hcsr_row_ptr;
        std::vector<int> hcsr_col_ind;
        std::vector<T>   hcsr_val;
        int              nnz;
        srand(12345ULL);
        if(binfile != "")
        {
            if(read_bin_matrix(
                   binfile.c_str(), m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, base)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else if(argus.laplacian)
        {
            m = n = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcsr_col_ind, hcsr_val, base);
            nnz   = hcsr_row_ptr[m];
        }
        else
        {
            std::vector<int> coo_row_ind;

            if(filename != "")
            {
                if(read_mtx_matrix(
                       filename.c_str(), m, n, nnz, coo_row_ind, hcsr_col_ind, hcsr_val, base)
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
                    scale = 2.0 / std::max(m, n);
                }
                nnz = m * scale * n;
                nnz = std::max(nnz, 1);
                gen_matrix_coo(m, n, nnz, coo_row_ind, hcsr_col_ind, hcsr_val, base);
            }

            // Convert COO to CSR
            hcsr_row_ptr.resize(m + 1, 0);
            for(int i = 0; i < nnz; ++i)
            {
                ++hcsr_row_ptr[coo_row_ind[i] + 1 - base];
            }

            hcsr_row_ptr[0] = base;
            for(int i = 0; i < m; ++i)
            {
                hcsr_row_ptr[i + 1] += hcsr_row_ptr[i];
            }
        }

        //
        // Now convert the CSR matrix to a GEBSR matrix.
        //
        bsr_mb         = m;
        bsr_nb         = n;
        bsr_nnzb       = nnz;
        size_t nvalues = bsr_nnzb * bsr_row_block_dim * bsr_col_block_dim;
        hbsr_val.resize(nvalues);
        for(size_t i = 0; i < nvalues; ++i)
        {
            hbsr_val[i] = random_generator<T>();
        }
        for(size_t i = 0; i < nvalues; ++i)
        {
            hbsr_val[i] = random_generator<T>();
        }
        hbsr_row_ptr = hcsr_row_ptr;
        hbsr_col_ind = hcsr_col_ind;
    }

    DEVICE_ALLOC(int, dbsr_row_ptr, (bsr_mb + 1));
    DEVICE_ALLOC(int, dbsr_col_ind, bsr_nnzb);
    DEVICE_ALLOC(T, dbsr_val, (bsr_nnzb * bsr_row_block_dim * bsr_col_block_dim));

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr.data(), sizeof(int) * (bsr_mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_col_ind, hbsr_col_ind.data(), sizeof(int) * bsr_nnzb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_val,
                              hbsr_val.data(),
                              sizeof(T) * bsr_nnzb * bsr_row_block_dim * bsr_col_block_dim,
                              hipMemcpyHostToDevice));

    //
    // Obtain required buffer size (from host)
    //
    size_t buffer_size;
    CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                              bsr_mb,
                                                              bsr_nb,
                                                              bsr_nnzb,
                                                              dbsr_val,
                                                              dbsr_row_ptr,
                                                              dbsr_col_ind,
                                                              bsr_row_block_dim,
                                                              bsr_col_block_dim,
                                                              &buffer_size));

    //
    // Allocate the buffer size.
    //
    auto  dbuffer_managed = hipsparse_unique_ptr{device_malloc(buffer_size), device_free};
    void* dbuffer         = dbuffer_managed.get();

    DEVICE_ALLOC(int, dbsc_row_ind, bsr_nnzb);
    DEVICE_ALLOC(int, dbsc_col_ptr, (bsr_nb + 1));
    DEVICE_ALLOC(T, dbsc_val, (bsr_nnzb * bsr_row_block_dim * bsr_col_block_dim));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsc<T>(handle,
                                                       bsr_mb,
                                                       bsr_nb,
                                                       bsr_nnzb,
                                                       dbsr_val,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind,
                                                       bsr_row_block_dim,
                                                       bsr_col_block_dim,
                                                       dbsc_val,
                                                       dbsc_row_ind,
                                                       dbsc_col_ptr,
                                                       action,
                                                       base,
                                                       dbuffer));
        //
        // Transfer to host.
        //
        std::vector<int> hbsc_from_device_row_ind(bsr_nnzb);
        std::vector<int> hbsc_from_device_col_ptr(bsr_nb + 1);
        std::vector<T>   hbsc_from_device_val(bsr_nnzb * bsr_row_block_dim * bsr_col_block_dim);
        CHECK_HIP_ERROR(hipMemcpy(hbsc_from_device_col_ptr.data(),
                                  dbsc_col_ptr,
                                  sizeof(int) * (bsr_nb + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsc_from_device_row_ind.data(),
                                  dbsc_row_ind,
                                  sizeof(int) * bsr_nnzb,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsc_from_device_val.data(),
                                  dbsc_val,
                                  sizeof(T) * bsr_nnzb * bsr_row_block_dim * bsr_col_block_dim,
                                  hipMemcpyDeviceToHost));

        //
        // Allocate host bsc matrix.
        //
        std::vector<int> hbsc_row_ind(bsr_nnzb);
        std::vector<int> hbsc_col_ptr(bsr_nb + 1);
        std::vector<T>   hbsc_val(bsr_nnzb * bsr_row_block_dim * bsr_col_block_dim);
        host_gebsr_to_gebsc<T>(bsr_mb,
                               bsr_nb,
                               bsr_nnzb,
                               hbsr_row_ptr,
                               hbsr_col_ind,
                               hbsr_val,
                               bsr_row_block_dim,
                               bsr_col_block_dim,
                               hbsc_row_ind,
                               hbsc_col_ptr,
                               hbsc_val,
                               action,
                               base);

        unit_check_general(1, bsr_nb + 1, 1, hbsc_from_device_col_ptr.data(), hbsc_col_ptr.data());
        unit_check_general(1, bsr_nnzb, 1, hbsc_from_device_row_ind.data(), hbsc_row_ind.data());
        if(action == HIPSPARSE_ACTION_NUMERIC)
        {
            unit_check_general(1,
                               bsr_nnzb * bsr_row_block_dim * bsr_col_block_dim,
                               1,
                               hbsc_from_device_val.data(),
                               hbsc_val.data());
        }
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_GEBSR2GEBSC_HPP
