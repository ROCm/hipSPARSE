/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_SDDMM_CSR_HPP
#define TESTING_SDDMM_CSR_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include "hipsparse_arguments.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_sddmm_csr_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVIDIA__
    // do not test for bad args
    return;
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11022)

    int32_t              n         = 100;
    int32_t              m         = 100;
    int32_t              k         = 100;
    int64_t              nnz       = 100;
    int32_t              safe_size = 100;
    float                alpha     = 0.6;
    float                beta      = 0.2;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOrder_t     order     = HIPSPARSE_ORDER_COL;
    hipsparseIndexBase_t idxBase   = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexType_t idxTypeI  = HIPSPARSE_INDEX_64I;
    hipsparseIndexType_t idxTypeJ  = HIPSPARSE_INDEX_32I;
    hipDataType          dataType  = HIP_R_32F;
    hipsparseSDDMMAlg_t  alg       = HIPSPARSE_SDDMM_ALG_DEFAULT;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int64_t) * safe_size), device_free};
    auto dcol_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int32_t) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dA_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int64_t* dptr = (int64_t*)dptr_managed.get();
    int32_t* dcol = (int32_t*)dcol_managed.get();
    float*   dval = (float*)dval_managed.get();
    float*   dB   = (float*)dB_managed.get();
    float*   dA   = (float*)dA_managed.get();
    void*    dbuf = (void*)dbuf_managed.get();

    if(!dval || !dptr || !dcol || !dB || !dA || !dbuf)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // SDDMM structures
    hipsparseDnMatDescr_t A, B;
    hipsparseSpMatDescr_t C;

    size_t bsize;

    // Create SDDMM structures
    verify_hipsparse_status_success(hipsparseCreateDnMat(&A, m, k, m, dA, dataType, order),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&B, k, n, k, dB, dataType, order),
                                    "success");
    verify_hipsparse_status_success(
        hipsparseCreateCsr(&C, m, n, nnz, dptr, dcol, dval, idxTypeI, idxTypeJ, idxBase, dataType),
        "success");

    // SDDMM buffer
    verify_hipsparse_status_invalid_handle(hipsparseSDDMM_bufferSize(
        nullptr, transA, transB, &alpha, A, B, &beta, C, dataType, alg, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_bufferSize(
            handle, transA, transB, nullptr, A, B, &beta, C, dataType, alg, &bsize),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_bufferSize(
            handle, transA, transB, &alpha, nullptr, B, &beta, C, dataType, alg, &bsize),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_bufferSize(
            handle, transA, transB, &alpha, A, nullptr, &beta, C, dataType, alg, &bsize),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_bufferSize(
            handle, transA, transB, &alpha, A, B, nullptr, C, dataType, alg, &bsize),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_bufferSize(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, &bsize),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_bufferSize(
            handle, transA, transB, &alpha, A, B, &beta, C, dataType, alg, nullptr),
        "Error: bsize is nullptr");

    // SDDMM
    verify_hipsparse_status_invalid_handle(hipsparseSDDMM_preprocess(
        nullptr, transA, transB, &alpha, A, B, &beta, C, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_preprocess(
            handle, transA, transB, nullptr, A, B, &beta, C, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_preprocess(
            handle, transA, transB, &alpha, nullptr, B, &beta, C, dataType, alg, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_preprocess(
            handle, transA, transB, &alpha, A, nullptr, &beta, C, dataType, alg, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_preprocess(
            handle, transA, transB, &alpha, A, B, nullptr, C, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_preprocess(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, dbuf),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM_preprocess(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, nullptr),
        "Error: dbuf is nullptr");

    // SDDMM
    verify_hipsparse_status_invalid_handle(
        hipsparseSDDMM(nullptr, transA, transB, &alpha, A, B, &beta, C, dataType, alg, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM(handle, transA, transB, nullptr, A, B, &beta, C, dataType, alg, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM(handle, transA, transB, &alpha, nullptr, B, &beta, C, dataType, alg, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM(handle, transA, transB, &alpha, A, nullptr, &beta, C, dataType, alg, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM(handle, transA, transB, &alpha, A, B, nullptr, C, dataType, alg, dbuf),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM(handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, dbuf),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSDDMM(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, nullptr),
        "Error: dbuf is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroyDnMat(A), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(B), "success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(C), "success");

#endif
}

template <typename I, typename J, typename T>
hipsparseStatus_t testing_sddmm_csr(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11022)
    J                    m        = argus.M;
    J                    n        = argus.N;
    J                    k        = argus.K;
    T                    h_alpha  = make_DataType<T>(argus.alpha);
    T                    h_beta   = make_DataType<T>(argus.beta);
    hipsparseOperation_t transA   = argus.transA;
    hipsparseOperation_t transB   = argus.transB;
    hipsparseOrder_t     order    = argus.orderA;
    hipsparseIndexBase_t idx_base = argus.baseA;
    hipsparseSDDMMAlg_t  alg      = HIPSPARSE_SDDMM_ALG_DEFAULT;
    std::string          filename = argus.filename;

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipsparseIndexType_t typeJ = getIndexType<J>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    // Host structures
    std::vector<I> hcsr_row_ptr;
    std::vector<J> hcsr_col_ind;
    std::vector<T> hcsr_val;

    // Initial Data on CPU
    srand(12345ULL);

    // Read or construct CSR matrix
    I nnz = 0;
    if(!generate_csr_matrix(filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    J lda = m;
    J ldb = k;

    std::vector<T> hA(m * k);
    std::vector<T> hB(k * n);
    std::vector<T> hval1(nnz);
    std::vector<T> hval2(nnz);

    hipsparseInit<T>(hA, m, k);
    hipsparseInit<T>(hB, k, n);

    // allocate memory on device
    auto dptr_managed  = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto dcol_managed  = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz), device_free};
    auto dval1_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dval2_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    auto dA_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * k), device_free};
    auto dB_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * k * n), device_free};

    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dptr  = (I*)dptr_managed.get();
    J* dcol  = (J*)dcol_managed.get();
    T* dval1 = (T*)dval1_managed.get();
    T* dval2 = (T*)dval2_managed.get();

    T* dA      = (T*)dA_managed.get();
    T* dB      = (T*)dB_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    T* d_beta  = (T*)d_beta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(I) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(J) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval1, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval2, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * m * k, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * k * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create matrices
    hipsparseSpMatDescr_t C1, C2;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCsr(&C1, m, n, nnz, dptr, dcol, dval1, typeI, typeJ, idx_base, typeT));
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCsr(&C2, m, n, nnz, dptr, dcol, dval2, typeI, typeJ, idx_base, typeT));

    // Create dense matrices
    hipsparseDnMatDescr_t A, B;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&A, m, k, lda, dA, typeT, order));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&B, k, n, ldb, dB, typeT, order));

    // Query SDDMM buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSDDMM_bufferSize(
        handle, transA, transB, &h_alpha, A, B, &h_beta, C1, typeT, alg, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // ROCSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseSDDMM_preprocess(
        handle, transA, transB, &h_alpha, A, B, &h_beta, C1, typeT, alg, buffer));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSDDMM(handle, transA, transB, &h_alpha, A, B, &h_beta, C1, typeT, alg, buffer));

    // ROCSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(hipsparseSDDMM_preprocess(
        handle, transA, transB, d_alpha, A, B, d_beta, C2, typeT, alg, buffer));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSDDMM(handle, transA, transB, d_alpha, A, B, d_beta, C2, typeT, alg, buffer));

    // copy output from device to CPU.
    CHECK_HIP_ERROR(hipMemcpy(hval1.data(), dval1, sizeof(T) * nnz, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hval2.data(), dval2, sizeof(T) * nnz, hipMemcpyDeviceToHost));

    // CPU
    const J incx = lda;
    const J incy = 1;

    for(J i = 0; i < m; ++i)
    {
        for(I at = hcsr_row_ptr[i] - idx_base; at < hcsr_row_ptr[i + 1] - idx_base; ++at)
        {
            J        j   = hcsr_col_ind[at] - idx_base;
            const T* x   = &hA[i];
            const T* y   = &hB[ldb * j];
            T        sum = make_DataType<T>(0.0);
            for(J k_ = 0; k_ < k; ++k_)
            {
                sum = testing_fma(x[incx * k_], y[incy * k_], sum);
            }
            hcsr_val[at] = testing_mult(hcsr_val[at], h_beta) + testing_mult(h_alpha, sum);
        }
    }

    unit_check_near(1, nnz, 1, hval1.data(), hcsr_val.data());
    unit_check_near(1, nnz, 1, hval2.data(), hcsr_val.data());

    // free.
    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(C1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(C2));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));

#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_SDDMM_CSR_HPP
