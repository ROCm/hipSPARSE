/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef TESTING_SPMM_COO_HPP
#define TESTING_SPMM_COO_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>
#include <unistd.h>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_spmm_coo_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVCC__
    // do not test for bad args
    return;
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
    int32_t              n         = 100;
    int32_t              m         = 100;
    int32_t              k         = 100;
    int32_t              ldb       = 100;
    int32_t              ldc       = 100;
    int32_t              nnz       = 100;
    int32_t              safe_size = 100;
    float                alpha     = 0.6;
    float                beta      = 0.2;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOrder_t     order     = HIPSPARSE_ORDER_COLUMN;
    hipsparseIndexBase_t idxBase   = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexType_t idxType   = HIPSPARSE_INDEX_32I;
    hipDataType          dataType  = HIP_R_32F;
    hipsparseSpMMAlg_t   alg       = HIPSPARSE_SPMM_COO_ALG1;
    hipsparseStatus_t    status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    auto drow_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int32_t) * safe_size), device_free};
    auto dcol_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int32_t) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dC_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int32_t* drow = (int32_t*)drow_managed.get();
    int32_t* dcol = (int32_t*)dcol_managed.get();
    float*   dval = (float*)dval_managed.get();
    float*   dB   = (float*)dB_managed.get();
    float*   dC   = (float*)dC_managed.get();
    void*    dbuf = (void*)dbuf_managed.get();

    if(!dval || !drow || !dcol || !dB || !dC || !dbuf)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // SpMM structures
    hipsparseSpMatDescr_t A;
    hipsparseDnMatDescr_t B, C;

    size_t bsize;

    // Create SpMM structures
    verify_hipsparse_status_success(
        hipsparseCreateCoo(&A, m, k, nnz, drow, dcol, dval, idxType, idxBase, dataType), "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&B, k, n, k, dB, dataType, order),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&C, m, n, m, dC, dataType, order),
                                    "success");

    // SpMM buffer
    verify_hipsparse_status_invalid_handle(hipsparseSpMM_bufferSize(
        nullptr, transA, transB, &alpha, A, B, &beta, C, dataType, alg, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, nullptr, A, B, &beta, C, dataType, alg, &bsize),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, nullptr, B, &beta, C, dataType, alg, &bsize),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, A, nullptr, &beta, C, dataType, alg, &bsize),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, A, B, nullptr, C, dataType, alg, &bsize),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, &bsize),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_bufferSize(
            handle, transA, transB, &alpha, A, B, &beta, C, dataType, alg, nullptr),
        "Error: bsize is nullptr");

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
    // SpMM_preprocess
    verify_hipsparse_status_invalid_handle(hipsparseSpMM_preprocess(
        nullptr, transA, transB, &alpha, A, B, &beta, C, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, nullptr, A, B, &beta, C, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, nullptr, B, &beta, C, dataType, alg, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, A, nullptr, &beta, C, dataType, alg, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, A, B, nullptr, C, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, dbuf),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM_preprocess(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, nullptr),
        "Error: dbuf is nullptr");
#endif

    // SpMM
    verify_hipsparse_status_invalid_handle(
        hipsparseSpMM(nullptr, transA, transB, &alpha, A, B, &beta, C, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(handle, transA, transB, nullptr, A, B, &beta, C, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(handle, transA, transB, &alpha, nullptr, B, &beta, C, dataType, alg, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(handle, transA, transB, &alpha, A, nullptr, &beta, C, dataType, alg, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(handle, transA, transB, &alpha, A, B, nullptr, C, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, dbuf),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpMM(handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, nullptr),
        "Error: dbuf is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpMat(A), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(B), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(C), "success");
#endif
}

template <typename I, typename T>
hipsparseStatus_t testing_spmm_coo()
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
    T                    h_alpha  = make_DataType<T>(2.0);
    T                    h_beta   = make_DataType<T>(1.0);
    hipsparseOperation_t transA   = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB   = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOrder_t     order    = HIPSPARSE_ORDER_COLUMN;
    hipsparseIndexBase_t idx_base = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseSpMMAlg_t   alg      = HIPSPARSE_SPMM_COO_ALG1;
    hipsparseStatus_t    status;

    // Determine absolute path of test matrix

    // Get current executables absolute path
    char    path_exe[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", path_exe, sizeof(path_exe) - 1);
    if(len < 14)
        path_exe[0] = '\0';
    else
        path_exe[len - 14] = '\0';

    // Matrices are stored at the same path in matrices directory
    std::string filename = std::string(path_exe) + "../matrices/nos3.bin";

    // Index and data type
    hipsparseIndexType_t typeI
        = (typeid(I) == typeid(int32_t)) ? HIPSPARSE_INDEX_32I : HIPSPARSE_INDEX_64I;
    hipDataType typeT = (typeid(T) == typeid(float))
                            ? HIP_R_32F
                            : ((typeid(T) == typeid(double))
                                   ? HIP_R_64F
                                   : ((typeid(T) == typeid(hipComplex) ? HIP_C_32F : HIP_C_64F)));

    // hipSPARSE handle
    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    // Host structures
    std::vector<I> hrow_ptr;
    std::vector<I> hcol_ind;
    std::vector<T> hval;

    // Initial Data on CPU
    srand(12345ULL);

    I m;
    I k;
    I nnz;

    if(read_bin_matrix(filename.c_str(), m, k, nnz, hrow_ptr, hcol_ind, hval, idx_base) != 0)
    {
        fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    std::vector<I> hrow_ind(nnz);

    // Convert to COO
    for(I i = 0; i < m; ++i)
    {
        for(I j = hrow_ptr[i]; j < hrow_ptr[i + 1]; ++j)
        {
            hrow_ind[j - idx_base] = i + idx_base;
        }
    }

    I n   = 5;
    I ldb = k;
    I ldc = m;

    std::vector<T> hB(k * n);
    std::vector<T> hC_1(m * n);
    std::vector<T> hC_2(m * n);
    std::vector<T> hC_gold(m * n);

    hipsparseInit<T>(hB, k, n);
    hipsparseInit<T>(hC_1, m, n);

    // copy vector is easy in STL; hC_gold = hB: save a copy in hy_gold which will be output of CPU
    hC_2    = hC_1;
    hC_gold = hC_1;

    // allocate memory on device
    auto drow_managed    = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dcol_managed    = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dval_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dB_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * k * n), device_free};
    auto dC_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * n), device_free};
    auto dC_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * n), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* drow    = (I*)drow_managed.get();
    I* dcol    = (I*)dcol_managed.get();
    T* dval    = (T*)dval_managed.get();
    T* dB      = (T*)dB_managed.get();
    T* dC_1    = (T*)dC_1_managed.get();
    T* dC_2    = (T*)dC_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    T* d_beta  = (T*)d_beta_managed.get();

    if(!dval || !drow || !dcol || !dB || !dC_1 || !dC_2 || !d_alpha || !d_beta)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dval || !drow || !dcol || !dB || "
                                        "!dC_1 || !dC_2 || !d_alpha || !d_beta");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(drow, hrow_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcol_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hval.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * k * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1.data(), sizeof(T) * m * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2.data(), sizeof(T) * m * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create matrices
    hipsparseSpMatDescr_t A;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCoo(&A, m, k, nnz, drow, dcol, dval, typeI, idx_base, typeT));

    // Create dense matrices
    hipsparseDnMatDescr_t B, C1, C2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&B, k, n, ldb, dB, typeT, order));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C1, m, n, ldc, dC_1, typeT, order));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C2, m, n, ldc, dC_2, typeT, order));

    // Query SpMM buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMM_bufferSize(
        handle, transA, transB, &h_alpha, A, B, &h_beta, C1, typeT, alg, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // ROCSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
    CHECK_HIPSPARSE_ERROR(hipsparseSpMM_preprocess(
        handle, transA, transB, &h_alpha, A, B, &h_beta, C1, typeT, alg, buffer));
#endif
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMM(handle, transA, transB, &h_alpha, A, B, &h_beta, C1, typeT, alg, buffer));

    // ROCSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
    CHECK_HIPSPARSE_ERROR(hipsparseSpMM_preprocess(
        handle, transA, transB, d_alpha, A, B, d_beta, C2, typeT, alg, buffer));
#endif
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMM(handle, transA, transB, d_alpha, A, B, d_beta, C2, typeT, alg, buffer));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC_1, sizeof(T) * m * n, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC_2, sizeof(T) * m * n, hipMemcpyDeviceToHost));

    // CPU
    double cpu_time_used = get_time_us();

    for(I j = 0; j < n; j++)
    {
        for(I i = 0; i < m; ++i)
        {
            I idx_C        = order == HIPSPARSE_ORDER_COLUMN ? i + j * ldc : i * ldc + j;
            hC_gold[idx_C] = h_beta * hC_gold[idx_C];
        }
    }

    for(I j = 0; j < n; j++)
    {
        for(I i = 0; i < nnz; ++i)
        {
            I row = hrow_ind[i] - idx_base;
            I col = hcol_ind[i] - idx_base;
            T val = h_alpha * hval[i];

            I idx_C = order == HIPSPARSE_ORDER_COLUMN ? row + j * ldc : row * ldc + j;

            I idx_B = 0;
            if((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE && order == HIPSPARSE_ORDER_COLUMN)
               || (transB == HIPSPARSE_OPERATION_TRANSPOSE && order == HIPSPARSE_ORDER_ROW))
            {
                idx_B = (col + j * ldb);
            }
            else
            {
                idx_B = (j + col * ldb);
            }

            hC_gold[idx_C] = hC_gold[idx_C] + val * hB[idx_B];
        }
    }

    cpu_time_used = get_time_us() - cpu_time_used;

    unit_check_near(1, m * n, 1, hC_gold.data(), hC_1.data());
    unit_check_near(1, m * n, 1, hC_gold.data(), hC_2.data());

    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C2));

#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_SPMM_COO_HPP
