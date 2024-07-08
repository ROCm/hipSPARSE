/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_SPMM_BATCHED_CSC_HPP
#define TESTING_SPMM_BATCHED_CSC_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_spmm_batched_csc_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVIDIA__
    // do not test for bad args
    return;
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
    int32_t              m         = 100;
    int32_t              n         = 100;
    int32_t              k         = 100;
    int64_t              nnz       = 100;
    float                alpha     = 0.6;
    float                beta      = 0.2;
    size_t               safe_size = 100;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOrder_t     order     = HIPSPARSE_ORDER_COL;
    hipsparseIndexBase_t idxBase   = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexType_t idxTypeI  = HIPSPARSE_INDEX_64I;
    hipsparseIndexType_t idxTypeJ  = HIPSPARSE_INDEX_32I;
    hipDataType          dataType  = HIP_R_32F;

#if(CUDART_VERSION >= 11003)
    hipsparseSpMMAlg_t alg = HIPSPARSE_SPMM_CSR_ALG1;
#else
    hipsparseSpMMAlg_t alg = HIPSPARSE_MM_ALG_DEFAULT;
#endif

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int64_t) * safe_size), device_free};
    auto drow_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int32_t) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dC_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int64_t* dptr = (int64_t*)dptr_managed.get();
    int32_t* drow = (int32_t*)drow_managed.get();
    float*   dval = (float*)dval_managed.get();
    float*   dB   = (float*)dB_managed.get();
    float*   dC   = (float*)dC_managed.get();
    void*    dbuf = (void*)dbuf_managed.get();

    if(!dval || !dptr || !drow || !dB || !dC || !dbuf)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // SpMM structures
    hipsparseSpMatDescr_t A;
    hipsparseDnMatDescr_t B, C;

    // Create SpMM structures
    verify_hipsparse_status_success(
        hipsparseCreateCsc(&A, m, k, nnz, dptr, drow, dval, idxTypeI, idxTypeJ, idxBase, dataType),
        "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&B, k, n, k, dB, dataType, order),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&C, m, n, m, dC, dataType, order),
                                    "success");

    int     batch_count_A;
    int     batch_count_B;
    int     batch_count_C;
    int64_t offsets_batch_stride_A;
    int64_t rows_values_batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;

    // C_i = A * B_i
    batch_count_A              = 1;
    batch_count_B              = 10;
    batch_count_C              = 5;
    offsets_batch_stride_A     = 0;
    rows_values_batch_stride_A = 0;
    batch_stride_B             = k * n;
    batch_stride_C             = m * n;
    verify_hipsparse_status_success(
        hipsparseCsrSetStridedBatch(
            A, batch_count_A, offsets_batch_stride_A, rows_values_batch_stride_A),
        "success");
    verify_hipsparse_status_success(hipsparseDnMatSetStridedBatch(B, batch_count_B, batch_stride_B),
                                    "success");
    verify_hipsparse_status_success(hipsparseDnMatSetStridedBatch(C, batch_count_C, batch_stride_C),
                                    "success");

    verify_hipsparse_status_invalid_value(
        hipsparseSpMM(handle, transA, transB, &alpha, A, B, &beta, C, dataType, alg, dbuf),
        "Error: Combination of strided batch parameters is invald");

    // C_i = A_i * B
    batch_count_A              = 10;
    batch_count_B              = 1;
    batch_count_C              = 5;
    offsets_batch_stride_A     = (k + 1);
    rows_values_batch_stride_A = nnz;
    batch_stride_B             = 0;
    batch_stride_C             = m * n;
    verify_hipsparse_status_success(
        hipsparseCsrSetStridedBatch(
            A, batch_count_A, offsets_batch_stride_A, rows_values_batch_stride_A),
        "success");
    verify_hipsparse_status_success(hipsparseDnMatSetStridedBatch(B, batch_count_B, batch_stride_B),
                                    "success");
    verify_hipsparse_status_success(hipsparseDnMatSetStridedBatch(C, batch_count_C, batch_stride_C),
                                    "success");

    verify_hipsparse_status_invalid_value(
        hipsparseSpMM(handle, transA, transB, &alpha, A, B, &beta, C, dataType, alg, dbuf),
        "Error: Combination of strided batch parameters is invald");

    // C_i = A_i * B_i
    batch_count_A              = 10;
    batch_count_B              = 10;
    batch_count_C              = 5;
    offsets_batch_stride_A     = (k + 1);
    rows_values_batch_stride_A = nnz;
    batch_stride_B             = k * n;
    batch_stride_C             = m * n;
    verify_hipsparse_status_success(
        hipsparseCsrSetStridedBatch(
            A, batch_count_A, offsets_batch_stride_A, rows_values_batch_stride_A),
        "success");
    verify_hipsparse_status_success(hipsparseDnMatSetStridedBatch(B, batch_count_B, batch_stride_B),
                                    "success");
    verify_hipsparse_status_success(hipsparseDnMatSetStridedBatch(C, batch_count_C, batch_stride_C),
                                    "success");

    verify_hipsparse_status_invalid_value(
        hipsparseSpMM(handle, transA, transB, &alpha, A, B, &beta, C, dataType, alg, dbuf),
        "Error: Combination of strided batch parameters is invald");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpMat(A), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(B), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(C), "success");
#endif
}

template <typename I, typename J, typename T>
hipsparseStatus_t testing_spmm_batched_csc(Arguments argus)
{
#ifdef __HIP_PLATFORM_NVIDIA__
    // do not test for bad args
    return HIPSPARSE_STATUS_SUCCESS;
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
    J                    m        = argus.M;
    J                    n        = argus.N;
    J                    k        = argus.K;
    T                    h_alpha  = make_DataType<T>(argus.alpha);
    T                    h_beta   = make_DataType<T>(argus.beta);
    hipsparseOperation_t transA   = argus.transA;
    hipsparseOperation_t transB   = argus.transB;
    hipsparseOrder_t     orderB   = argus.orderB;
    hipsparseOrder_t     orderC   = argus.orderC;
    hipsparseIndexBase_t idx_base = argus.idx_base;

    J batch_count_A = 1;
    J batch_count_B = 3;
    J batch_count_C = 3;

#if(CUDART_VERSION >= 11003)
    hipsparseSpMMAlg_t alg = HIPSPARSE_SPMM_CSR_ALG1;
#else
    hipsparseSpMMAlg_t alg = HIPSPARSE_MM_ALG_DEFAULT;
#endif

    // Matrices are stored at the same path in matrices directory
    std::string filename = get_filename("nos3.bin");

#if(defined(CUDART_VERSION))
    if(orderB != orderC || orderB != HIPSPARSE_ORDER_COL)
    {
        return HIPSPARSE_STATUS_SUCCESS;
    }
#endif

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipsparseIndexType_t typeJ = getIndexType<J>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    // Host structures
    std::vector<I> hcsc_col_ptr_temp;
    std::vector<J> hcsc_row_ind_temp;
    std::vector<T> hcsc_val_temp;

    // Initial Data on CPU
    srand(12345ULL);

    I nnz_A;
    if(!generate_csr_matrix(filename,
                            (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : m,
                            (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : k,
                            nnz_A,
                            hcsc_col_ptr_temp,
                            hcsc_row_ind_temp,
                            hcsc_val_temp,
                            idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // Some matrix properties
    J A_m = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : k;
    J A_n = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : m;
    J B_m = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : n;
    J B_n = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? n : k;
    J C_m = m;
    J C_n = n;

    int ld_multiplier_B = 1;
    int ld_multiplier_C = 1;

    int64_t ldb
        = (orderB == HIPSPARSE_ORDER_COL)
              ? ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? (int64_t(ld_multiplier_B) * k)
                                                               : (int64_t(ld_multiplier_B) * n))
              : ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? (int64_t(ld_multiplier_B) * n)
                                                               : (int64_t(ld_multiplier_B) * k));
    int64_t ldc = (orderC == HIPSPARSE_ORDER_COL) ? (int64_t(ld_multiplier_C) * m)
                                                  : (int64_t(ld_multiplier_C) * n);

    ldb = std::max(int64_t(1), ldb);
    ldc = std::max(int64_t(1), ldc);

    int64_t nrowB = (orderB == HIPSPARSE_ORDER_COL) ? ldb : B_m;
    int64_t ncolB = (orderB == HIPSPARSE_ORDER_COL) ? B_n : ldb;
    int64_t nrowC = (orderC == HIPSPARSE_ORDER_COL) ? ldc : C_m;
    int64_t ncolC = (orderC == HIPSPARSE_ORDER_COL) ? C_n : ldc;

    int64_t nnz_B = nrowB * ncolB;
    int64_t nnz_C = nrowC * ncolC;

    int64_t offsets_batch_stride_A     = (batch_count_A > 1) ? (A_n + 1) : 0;
    int64_t rows_values_batch_stride_A = (batch_count_A > 1) ? nnz_A : 0;
    int64_t batch_stride_B             = (batch_count_B > 1) ? nnz_B : 0;
    int64_t batch_stride_C             = (batch_count_C > 1) ? nnz_C : 0;

    // Allocate host memory for all batches of A matrix
    std::vector<I> hcsc_col_ptr(batch_count_A * (A_n + 1));
    std::vector<J> hcsc_row_ind(batch_count_A * nnz_A);
    std::vector<T> hcsc_val(batch_count_A * nnz_A);

    for(J i = 0; i < batch_count_A; i++)
    {
        for(J j = 0; j < (A_n + 1); j++)
        {
            hcsc_col_ptr[(A_n + 1) * i + j] = hcsc_col_ptr_temp[j];
        }

        for(I j = 0; j < nnz_A; j++)
        {
            hcsc_row_ind[nnz_A * i + j] = hcsc_row_ind_temp[j];
            hcsc_val[nnz_A * i + j]     = hcsc_val_temp[j];
        }
    }

    // Allocate host memory for vectors
    std::vector<T> hB(batch_count_B * nnz_B);
    std::vector<T> hC_1(batch_count_C * nnz_C);
    std::vector<T> hC_2(batch_count_C * nnz_C);
    std::vector<T> hC_gold(batch_count_C * nnz_C);

    hipsparseInit<T>(hB, batch_count_B * nnz_B, 1);
    hipsparseInit<T>(hC_1, batch_count_C * nnz_C, 1);

    // copy vector is easy in STL; hC_gold = hC: save a copy in hy_gold which will be output of CPU
    hC_2    = hC_1;
    hC_gold = hC_1;

    // allocate memory on device
    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * (A_n + 1)), device_free};
    auto drow_managed = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz_A), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dB_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * batch_count_B * nnz_B), device_free};
    auto dC_1_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * batch_count_C * nnz_C), device_free};
    auto dC_2_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * batch_count_C * nnz_C), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dptr    = (I*)dptr_managed.get();
    J* drow    = (J*)drow_managed.get();
    T* dval    = (T*)dval_managed.get();
    T* dB      = (T*)dB_managed.get();
    T* dC_1    = (T*)dC_1_managed.get();
    T* dC_2    = (T*)dC_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    T* d_beta  = (T*)d_beta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsc_col_ptr.data(), sizeof(I) * (A_n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(drow, hcsc_row_ind.data(), sizeof(J) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsc_val.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dB, hB.data(), sizeof(T) * batch_count_B * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dC_1, hC_1.data(), sizeof(T) * batch_count_C * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dC_2, hC_2.data(), sizeof(T) * batch_count_C * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create matrices
    hipsparseSpMatDescr_t A;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCsc(&A, A_m, A_n, nnz_A, dptr, drow, dval, typeI, typeJ, idx_base, typeT));

    // Create dense matrices
    hipsparseDnMatDescr_t B, C1, C2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&B, B_m, B_n, ldb, dB, typeT, orderB));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C1, C_m, C_n, ldc, dC_1, typeT, orderC));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C2, C_m, C_n, ldc, dC_2, typeT, orderC));

    CHECK_HIPSPARSE_ERROR(hipsparseCsrSetStridedBatch(
        A, batch_count_A, offsets_batch_stride_A, rows_values_batch_stride_A));
    CHECK_HIPSPARSE_ERROR(hipsparseDnMatSetStridedBatch(B, batch_count_B, batch_stride_B));
    CHECK_HIPSPARSE_ERROR(hipsparseDnMatSetStridedBatch(C1, batch_count_C, batch_stride_C));
    CHECK_HIPSPARSE_ERROR(hipsparseDnMatSetStridedBatch(C2, batch_count_C, batch_stride_C));

    // Query SpMM buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMM_bufferSize(
        handle, transA, transB, &h_alpha, A, B, &h_beta, C1, typeT, alg, &bufferSize));

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
    //When using cusparse backend, cant pass nullptr for buffer to preprocess
    if(bufferSize == 0)
    {
        bufferSize = 4;
    }
#endif

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
    CHECK_HIP_ERROR(
        hipMemcpy(hC_1.data(), dC_1, sizeof(T) * batch_count_C * nnz_C, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hC_2.data(), dC_2, sizeof(T) * batch_count_C * nnz_C, hipMemcpyDeviceToHost));

    // CPU
    double cpu_time_used = get_time_us();

    host_cscmm_batched(A_m,
                       n,
                       A_n,
                       batch_count_A,
                       (I)offsets_batch_stride_A,
                       (I)rows_values_batch_stride_A,
                       transA,
                       transB,
                       h_alpha,
                       hcsc_col_ptr.data(),
                       hcsc_row_ind.data(),
                       hcsc_val.data(),
                       hB.data(),
                       (J)ldb,
                       batch_count_B,
                       (I)batch_stride_B,
                       orderB,
                       h_beta,
                       hC_gold.data(),
                       (J)ldc,
                       batch_count_C,
                       (I)batch_stride_C,
                       orderC,
                       idx_base);

    cpu_time_used = get_time_us() - cpu_time_used;

    unit_check_near(1, batch_count_C * nnz_C, 1, hC_gold.data(), hC_1.data());
    unit_check_near(1, batch_count_C * nnz_C, 1, hC_gold.data(), hC_2.data());

    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C2));

#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_SPMM_BATCHED_CSC_HPP
