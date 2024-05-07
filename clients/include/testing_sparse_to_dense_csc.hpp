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
#ifndef TESTING_SPARSE_TO_DENSE_CSC_HPP
#define TESTING_SPARSE_TO_DENSE_CSC_HPP

#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse_test;

void testing_sparse_to_dense_csc_bad_arg(void)
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
    hipsparseIndexType_t iType    = HIPSPARSE_INDEX_64I;
    hipsparseIndexType_t jType    = HIPSPARSE_INDEX_32I;
    hipDataType          dataType = HIP_R_32F;

    // Create handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto ddense_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dcsc_col_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int64_t) * safe_size), device_free};
    auto dcsc_row_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int32_t) * safe_size), device_free};
    auto dcsc_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    float*   ddense_val   = (float*)ddense_val_managed.get();
    int64_t* dcsc_col_ptr = (int64_t*)dcsc_col_ptr_managed.get();
    int32_t* dcsc_row_ind = (int32_t*)dcsc_row_ind_managed.get();
    float*   dcsc_val     = (float*)dcsc_val_managed.get();
    void*    dbuf         = (void*)dbuf_managed.get();

    if(!ddense_val || !dcsc_col_ptr || !dcsc_row_ind || !dcsc_val || !dbuf)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Matrix structures
    hipsparseSpMatDescr_t matA;
    hipsparseDnVecDescr_t matB;

    size_t bsize;

    // Create matrix structures
    verify_hipsparse_status_success(hipsparseCreateCsr(&matA,
                                                       m,
                                                       n,
                                                       nnz,
                                                       dcsc_col_ptr,
                                                       dcsc_row_ind,
                                                       dcsc_val,
                                                       iType,
                                                       jType,
                                                       idxBase,
                                                       dataType),
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
    verify_hipsparse_status_invalid_pointer(
        hipsparseSparseToDense(handle, matA, matB, alg, nullptr), "Error: dbuf is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpMat(matA), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(matB), "success");
#endif
}

template <typename I, typename J, typename T>
hipsparseStatus_t testing_sparse_to_dense_csc(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
    hipsparseIndexBase_t        idx_base = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseSparseToDenseAlg_t alg      = HIPSPARSE_SPARSETODENSE_ALG_DEFAULT;
    hipsparseOrder_t            order    = HIPSPARSE_ORDER_COL;

    // Matrices are stored at the same path in matrices directory
    std::string filename = get_filename("nos3.bin");

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipsparseIndexType_t typeJ = getIndexType<J>();
    hipDataType typeT = getDataType<T>();
    //hipsparseIndexType_t typeI
    //    = (typeid(I) == typeid(int32_t)) ? HIPSPARSE_INDEX_32I : HIPSPARSE_INDEX_64I;
    //hipsparseIndexType_t typeJ
    //    = (typeid(J) == typeid(int32_t)) ? HIPSPARSE_INDEX_32I : HIPSPARSE_INDEX_64I;
    //hipDataType typeT = (typeid(T) == typeid(float))
    //                        ? HIP_R_32F
    //                        : ((typeid(T) == typeid(double))
    //                               ? HIP_R_64F
    //                               : ((typeid(T) == typeid(hipComplex) ? HIP_C_32F : HIP_C_64F)));

    // hipSPARSE handle
    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    // Host structures for CSR matrix
    std::vector<I> hcsr_row_ptr;
    std::vector<J> hcsr_col_ind;
    std::vector<T> hcsr_val;

    // Initial Data on CPU
    srand(12345ULL);

    J m;
    J n;
    I nnz;

    if(read_bin_matrix(filename.c_str(), m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base)
       != 0)
    {
        fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    I ld = m;

    // Convert CSR matrix to CSC
    std::vector<I> hcsc_col_ptr(n + 1);
    std::vector<J> hcsc_row_ind(nnz);
    std::vector<T> hcsc_val(nnz);

    // Determine nnz per column
    for(I i = 0; i < nnz; ++i)
    {
        ++hcsc_col_ptr[hcsr_col_ind[i] + 1 - idx_base];
    }

    // Scan
    for(J i = 0; i < n; ++i)
    {
        hcsc_col_ptr[i + 1] += hcsc_col_ptr[i];
    }

    // Fill row indices and values
    for(J i = 0; i < m; ++i)
    {
        for(I j = hcsr_row_ptr[i]; j < hcsr_row_ptr[i + 1]; ++j)
        {
            J col = hcsr_col_ind[j - idx_base] - idx_base;
            I idx = hcsc_col_ptr[col];

            hcsc_row_ind[idx] = i + idx_base;
            hcsc_val[idx]     = hcsr_val[j - idx_base];

            ++hcsc_col_ptr[col];
        }
    }

    // Shift column pointer array
    for(J i = n; i > 0; --i)
    {
        hcsc_col_ptr[i] = hcsc_col_ptr[i - 1] + idx_base;
    }

    hcsc_col_ptr[0] = idx_base;

    // allocate memory on device
    auto dptr_managed   = hipsparse_unique_ptr{device_malloc(sizeof(I) * (n + 1)), device_free};
    auto drow_managed   = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz), device_free};
    auto dval_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto ddense_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * ld * n), device_free};

    I* dptr   = (I*)dptr_managed.get();
    J* drow   = (J*)drow_managed.get();
    T* dval   = (T*)dval_managed.get();
    T* ddense = (T*)ddense_managed.get();

    //if(!dval || !dptr || !drow || !ddense)
    //{
    //    verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
    //                                    "!dval || !dptr || !drow || !ddense");
    //    return HIPSPARSE_STATUS_ALLOC_FAILED;
    //}

    // Dense matrix
    std::vector<T> hdense(ld * n);

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsc_col_ptr.data(), sizeof(I) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(drow, hcsc_row_ind.data(), sizeof(J) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsc_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Create matrices
    hipsparseSpMatDescr_t matA;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCsc(&matA, m, n, nnz, dptr, drow, dval, typeI, typeJ, idx_base, typeT));

    // Create dense matrix
    hipsparseDnMatDescr_t matB;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&matB, m, n, ld, ddense, typeT, order));

    // Query SparseToDense buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSparseToDense_bufferSize(handle, matA, matB, alg, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    CHECK_HIPSPARSE_ERROR(hipsparseSparseToDense(handle, matA, matB, alg, buffer));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hdense.data(), ddense, sizeof(T) * ld * n, hipMemcpyDeviceToHost));

    // Query for warpSize
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);

    std::vector<T> hdense_cpu(ld * n);

    for(J col = 0; col < n; ++col)
    {
        for(J row = 0; row < m; ++row)
        {
            hdense_cpu[row + ld * col] = make_DataType<T>(0.0);
        }
    }

    for(J col = 0; col < n; ++col)
    {
        I start = hcsc_col_ptr[col] - idx_base;
        I end   = hcsc_col_ptr[col + 1] - idx_base;

        for(I at = start; at < end; ++at)
        {
            hdense_cpu[(hcsc_row_ind[at] - idx_base) + ld * col] = hcsc_val[at];
        }
    }

    unit_check_general(m, n, ld, hdense_cpu.data(), hdense.data());

    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matA));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(matB));
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_SPARSE_TO_DENSE_CSC_HPP
