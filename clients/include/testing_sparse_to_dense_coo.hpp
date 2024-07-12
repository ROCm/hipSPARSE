/* ************************************************************************
 * Copyright (C) 2020 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_SPARSE_TO_DENSE_COO_HPP
#define TESTING_SPARSE_TO_DENSE_COO_HPP

#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse_test;

void testing_sparse_to_dense_coo_bad_arg(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
    int64_t safe_size = 100;
    int32_t m         = 10;
    int32_t n         = 10;
    int64_t nnz       = 10;
    int64_t ld        = m;

    hipsparseIndexBase_t        idxBase = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseSparseToDenseAlg_t alg     = HIPSPARSE_SPARSETODENSE_ALG_DEFAULT;
    hipsparseOrder_t            order   = HIPSPARSE_ORDER_COL;

    // Index and data type
    hipsparseIndexType_t iType    = HIPSPARSE_INDEX_32I;
    hipDataType          dataType = HIP_R_32F;

    // Create handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto ddense_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dcoo_row_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int32_t) * safe_size), device_free};
    auto dcoo_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int32_t) * safe_size), device_free};
    auto dcoo_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    float*   ddense_val   = (float*)ddense_val_managed.get();
    int32_t* dcoo_row_ind = (int32_t*)dcoo_row_ind_managed.get();
    int32_t* dcoo_col_ind = (int32_t*)dcoo_col_ind_managed.get();
    float*   dcoo_val     = (float*)dcoo_val_managed.get();
    void*    dbuf         = (void*)dbuf_managed.get();

    // Matrix structures
    hipsparseSpMatDescr_t matA;
    hipsparseDnVecDescr_t matB;

    size_t bsize;

    // Create matrix structures
    verify_hipsparse_status_success(
        hipsparseCreateCoo(
            &matA, m, n, nnz, dcoo_row_ind, dcoo_col_ind, dcoo_val, iType, idxBase, dataType),
        "success");
    verify_hipsparse_status_success(
        hipsparseCreateDnMat(&matB, m, n, ld, ddense_val, dataType, order), "success");

    // SparseToDense buffer size
    verify_hipsparse_status_invalid_handle(
        hipsparseSparseToDense_bufferSize(nullptr, matA, matB, alg, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSparseToDense_bufferSize(handle, nullptr, matB, alg, &bsize),
        "Error: matA is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSparseToDense_bufferSize(handle, matA, nullptr, alg, &bsize),
        "Error: matB is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSparseToDense_bufferSize(handle, matA, matB, alg, nullptr),
        "Error: bsize is nullptr");

    // SparseToDense
    verify_hipsparse_status_invalid_handle(hipsparseSparseToDense(nullptr, matA, matB, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSparseToDense(handle, nullptr, matB, alg, dbuf), "Error: matA is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSparseToDense(handle, matA, nullptr, alg, dbuf), "Error: matB is nullptr");
    // cuda returns success here
#if(!defined(CUDART_VERSION))
    verify_hipsparse_status_invalid_pointer(
        hipsparseSparseToDense(handle, matA, matB, alg, nullptr), "Error: dbuf is nullptr");
#endif

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpMat(matA), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(matB), "success");
#endif
}

template <typename I, typename T>
hipsparseStatus_t testing_sparse_to_dense_coo(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
    I                           m        = argus.M;
    I                           n        = argus.N;
    hipsparseOrder_t            order    = argus.orderA;
    hipsparseIndexBase_t        idx_base = argus.baseA;
    hipsparseSparseToDenseAlg_t alg      = static_cast<hipsparseSparseToDenseAlg_t>(argus.sparse2dense_alg);
    std::string                 filename = argus.filename;

    std::cout << "m: " << m << " n: " << n << " order: " << order << " idx_base: " << idx_base
              << " alg: " << alg << " filename: " << filename << std::endl;

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    // Host structures
    std::vector<I> hcsr_row_ptr;
    std::vector<I> hcsr_col_ind;
    std::vector<T> hcsr_val;

    // Initial Data on CPU
    srand(12345ULL);

    I nnz;
    if(!generate_csr_matrix(filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    I ld = (order == HIPSPARSE_ORDER_COL) ? m : n;

    I nrows = (order == HIPSPARSE_ORDER_COL) ? ld : m;
    I ncols = (order == HIPSPARSE_ORDER_COL) ? n : ld;

    // Fill host COO arrays
    std::vector<I> hcoo_row_ind(nnz);
    std::vector<I> hcoo_col_ind = hcsr_col_ind;
    std::vector<T> hcoo_val     = hcsr_val;

    for(I i = 0; i < m; i++)
    {
        I start = hcsr_row_ptr[i] - idx_base;
        I end   = hcsr_row_ptr[i + 1] - idx_base;

        for(I j = start; j < end; j++)
        {
            hcoo_row_ind[j] = i + idx_base;
        }
    }

    // allocate memory on device
    auto drow_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto ddense_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * nrows * ncols), device_free};

    I* drow   = (I*)drow_managed.get();
    I* dcol   = (I*)dcol_managed.get();
    T* dval   = (T*)dval_managed.get();
    T* ddense = (T*)ddense_managed.get();

    // Dense matrix
    std::vector<T> hdense(nrows * ncols);

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(drow, hcoo_row_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcoo_col_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcoo_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Create matrices
    hipsparseSpMatDescr_t matA;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCoo(&matA, m, n, nnz, drow, dcol, dval, typeI, idx_base, typeT));

    // Create dense matrix
    hipsparseDnMatDescr_t matB;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&matB, m, n, ld, ddense, typeT, order));

    // Query SparseToDense buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSparseToDense_bufferSize(handle, matA, matB, alg, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseSparseToDense(handle, matA, matB, alg, buffer));

        // copy output from device to CPU
        CHECK_HIP_ERROR(
            hipMemcpy(hdense.data(), ddense, sizeof(T) * nrows * ncols, hipMemcpyDeviceToHost));

        std::vector<T> hdense_cpu(nrows * ncols);

        if(order == HIPSPARSE_ORDER_COL)
        {
            for(I col = 0; col < n; ++col)
            {
                for(I row = 0; row < m; ++row)
                {
                    hdense_cpu[row + ld * col] = make_DataType<T>(0.0);
                }
            }

            for(I i = 0; i < nnz; i++)
            {
                I row = hcoo_row_ind[i] - idx_base;
                I col = hcoo_col_ind[i] - idx_base;

                hdense_cpu[ld * col + row] = hcoo_val[i];
            }
        }
        else
        {
            for(I row = 0; row < m; ++row)
            {
                for(I col = 0; col < n; ++col)
                {
                    hdense_cpu[ld * row + col] = make_DataType<T>(0.0);
                }
            }

            for(I i = 0; i < nnz; i++)
            {
                I row = hcoo_row_ind[i] - idx_base;
                I col = hcoo_col_ind[i] - idx_base;

                hdense_cpu[col + ld * row] = hcoo_val[i];
            }
        }

        unit_check_general(1, nrows * ncols, 1, hdense_cpu.data(), hdense.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm-up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseSparseToDense(handle, matA, matB, alg, buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseSparseToDense(handle, matA, matB, alg, buffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = coo2dense_gbyte_count<T>(m, n, (I)nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        std::cout << "GBytes/s: " << gpu_gbyte << " time (ms): " << get_gpu_time_msec(gpu_time_used)
                  << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matA));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(matB));
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_SPARSE_TO_DENSE_COO_HPP
