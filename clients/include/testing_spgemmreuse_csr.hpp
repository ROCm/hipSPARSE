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
#ifndef TESTING_SPGEMMREUSE_CSR_HPP
#define TESTING_SPGEMMREUSE_CSR_HPP

#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse_test;

void testing_spgemmreuse_csr_bad_arg(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
    int64_t              m         = 100;
    int64_t              n         = 100;
    int64_t              k         = 100;
    int64_t              nnz_A     = 100;
    int64_t              nnz_B     = 100;
    int64_t              nnz_C     = 100;
    int64_t              safe_size = 100;
    float                alpha     = 0.6;
    float                beta      = 0.2;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idxBaseA  = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t idxBaseB  = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t idxBaseC  = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexType_t idxType   = HIPSPARSE_INDEX_32I;
    hipDataType          dataType  = HIP_R_32F;
    hipsparseSpGEMMAlg_t alg       = HIPSPARSE_SPGEMM_DEFAULT;
    hipsparseStatus_t    status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<spgemm_struct> unique_ptr_descr(new spgemm_struct);
    hipsparseSpGEMMDescr_t         descr = unique_ptr_descr->descr;
    auto                           dcsr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcsr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcsr_val_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dcsr_row_ptr_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcsr_col_ind_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcsr_val_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dcsr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcsr_col_ind_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcsr_val_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*   dcsr_row_ptr_A = (int*)dcsr_row_ptr_A_managed.get();
    int*   dcsr_col_ind_A = (int*)dcsr_col_ind_A_managed.get();
    float* dcsr_val_A     = (float*)dcsr_val_A_managed.get();
    int*   dcsr_row_ptr_B = (int*)dcsr_row_ptr_B_managed.get();
    int*   dcsr_col_ind_B = (int*)dcsr_col_ind_B_managed.get();
    float* dcsr_val_B     = (float*)dcsr_val_B_managed.get();
    int*   dcsr_row_ptr_C = (int*)dcsr_row_ptr_C_managed.get();
    int*   dcsr_col_ind_C = (int*)dcsr_col_ind_C_managed.get();
    float* dcsr_val_C     = (float*)dcsr_val_C_managed.get();
    void*  dbuf           = (void*)dbuf_managed.get();

    if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_B || !dcsr_col_ind_B
       || !dcsr_val_B || !dcsr_row_ptr_C || !dcsr_col_ind_C || !dcsr_val_C || !dbuf)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // SpGEMM structures
    hipsparseSpMatDescr_t A, B, C;

    size_t bufferSize;

    // Create SpGEMM structures
    verify_hipsparse_status_success(hipsparseCreateCsr(&A,
                                                       m,
                                                       k,
                                                       nnz_A,
                                                       dcsr_row_ptr_A,
                                                       dcsr_col_ind_A,
                                                       dcsr_val_A,
                                                       idxType,
                                                       idxType,
                                                       idxBaseA,
                                                       dataType),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateCsr(&B,
                                                       k,
                                                       n,
                                                       nnz_B,
                                                       dcsr_row_ptr_B,
                                                       dcsr_col_ind_B,
                                                       dcsr_val_B,
                                                       idxType,
                                                       idxType,
                                                       idxBaseB,
                                                       dataType),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateCsr(&C,
                                                       m,
                                                       n,
                                                       nnz_C,
                                                       dcsr_row_ptr_C,
                                                       dcsr_col_ind_C,
                                                       dcsr_val_C,
                                                       idxType,
                                                       idxType,
                                                       idxBaseC,
                                                       dataType),
                                    "success");

    // SpGEMM work estimation
    verify_hipsparse_status_invalid_handle(hipsparseSpGEMMreuse_workEstimation(
        nullptr, transA, transB, A, B, C, alg, descr, &bufferSize, nullptr));

    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEMMreuse_workEstimation(
            handle, transA, transB, nullptr, B, C, alg, descr, &bufferSize, nullptr),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEMMreuse_workEstimation(
            handle, transA, transB, A, nullptr, C, alg, descr, &bufferSize, nullptr),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEMMreuse_workEstimation(
            handle, transA, transB, A, B, nullptr, alg, descr, &bufferSize, nullptr),
        "Error: C is nullptr");

    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEMMreuse_workEstimation(
            handle, transA, transB, A, B, C, alg, descr, nullptr, nullptr),
        "Error: bufferSize is nullptr");

    // SpGEMM compute
    verify_hipsparse_status_invalid_handle(hipsparseSpGEMMreuse_compute(
        nullptr, transA, transB, &alpha, A, B, &beta, C, dataType, alg, descr));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEMMreuse_compute(
            handle, transA, transB, nullptr, A, B, &beta, C, dataType, alg, descr),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEMMreuse_compute(
            handle, transA, transB, &alpha, nullptr, B, &beta, C, dataType, alg, descr),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEMMreuse_compute(
            handle, transA, transB, &alpha, A, nullptr, &beta, C, dataType, alg, descr),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEMMreuse_compute(
            handle, transA, transB, &alpha, A, B, nullptr, C, dataType, alg, descr),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEMMreuse_compute(
            handle, transA, transB, &alpha, A, B, &beta, nullptr, dataType, alg, descr),
        "Error: C is nullptr");

    // SpGEMMreuse copy
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEMMreuse_copy(
            handle, transA, transB, nullptr, B, C, alg, descr, &bufferSize, nullptr),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEMMreuse_copy(
            handle, transA, transB, A, nullptr, C, alg, descr, &bufferSize, nullptr),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEMMreuse_copy(
            handle, transA, transB, A, B, nullptr, alg, descr, &bufferSize, nullptr),
        "Error: C is nullptr");

    verify_hipsparse_status_invalid_pointer(
        hipsparseSpGEMMreuse_copy(handle, transA, transB, A, B, C, alg, descr, nullptr, nullptr),
        "Error: bufferSize is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpMat(A), "success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(B), "success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(C), "success");
#endif
}

template <typename I, typename J, typename T>
hipsparseStatus_t testing_spgemmreuse_csr(void)
{

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
    I                    safe_size = 100;
    T                    h_alpha   = make_DataType<T>(2.0);
    T                    h_beta    = make_DataType<T>(0.0);
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idxBaseA  = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t idxBaseB  = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t idxBaseC  = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseSpGEMMAlg_t alg       = HIPSPARSE_SPGEMM_DEFAULT;
    hipsparseStatus_t    status;

    // Matrices are stored at the same path in matrices directory
    std::string filename = hipsparse_exepath() + "../matrices/nos6.bin";

    // Index and data type
    hipsparseIndexType_t typeI
        = (typeid(I) == typeid(int32_t)) ? HIPSPARSE_INDEX_32I : HIPSPARSE_INDEX_64I;
    hipsparseIndexType_t typeJ
        = (typeid(J) == typeid(int32_t)) ? HIPSPARSE_INDEX_32I : HIPSPARSE_INDEX_64I;
    hipDataType typeT = (typeid(T) == typeid(float))
                            ? HIP_R_32F
                            : ((typeid(T) == typeid(double))
                                   ? HIP_R_64F
                                   : ((typeid(T) == typeid(hipComplex) ? HIP_C_32F : HIP_C_64F)));

    // hipSPARSE handles
    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    std::unique_ptr<spgemm_struct> unique_ptr_descr(new spgemm_struct);
    hipsparseSpGEMMDescr_t         descr = unique_ptr_descr->descr;

    // Host structures
    std::vector<I> hcsr_row_ptr_A;
    std::vector<J> hcsr_col_ind_A;
    std::vector<T> hcsr_val_A;

    // Initial Data on CPU
    srand(12345ULL);

    // Some sparse matrix A
    J m;
    J k;
    I nnz_A;

    if(read_bin_matrix(
           filename.c_str(), m, k, nnz_A, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idxBaseA)
       != 0)
    {
        fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // Sparse matrix B as the transpose of A
    J n     = m;
    I nnz_B = nnz_A;

    std::vector<I> hcsr_row_ptr_B(k + 1);
    std::vector<J> hcsr_col_ind_B(nnz_B);
    std::vector<T> hcsr_val_B(nnz_B);

    transpose_csr(m,
                  k,
                  nnz_A,
                  hcsr_row_ptr_A.data(),
                  hcsr_col_ind_A.data(),
                  hcsr_val_A.data(),
                  hcsr_row_ptr_B.data(),
                  hcsr_col_ind_B.data(),
                  hcsr_val_B.data(),
                  idxBaseA,
                  idxBaseB);

    // allocate memory on device
    auto dcsr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto dcsr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz_A), device_free};
    auto dcsr_val_A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dcsr_row_ptr_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(I) * (k + 1)), device_free};
    auto dcsr_col_ind_B_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz_B), device_free};
    auto dcsr_val_B_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dcsr_row_ptr_C_1_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto dcsr_row_ptr_C_2_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dcsr_row_ptr_A   = (I*)dcsr_row_ptr_A_managed.get();
    J* dcsr_col_ind_A   = (J*)dcsr_col_ind_A_managed.get();
    T* dcsr_val_A       = (T*)dcsr_val_A_managed.get();
    I* dcsr_row_ptr_B   = (I*)dcsr_row_ptr_B_managed.get();
    J* dcsr_col_ind_B   = (J*)dcsr_col_ind_B_managed.get();
    T* dcsr_val_B       = (T*)dcsr_val_B_managed.get();
    I* dcsr_row_ptr_C_1 = (I*)dcsr_row_ptr_C_1_managed.get();
    I* dcsr_row_ptr_C_2 = (I*)dcsr_row_ptr_C_2_managed.get();
    T* d_alpha          = (T*)d_alpha_managed.get();
    T* d_beta           = (T*)d_beta_managed.get();

    if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_B || !dcsr_col_ind_B
       || !dcsr_val_B || !dcsr_row_ptr_C_1 || !dcsr_row_ptr_C_2 || !d_alpha || !d_beta)
    {
        verify_hipsparse_status_success(
            HIPSPARSE_STATUS_ALLOC_FAILED,
            "!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || "
            "!dcsr_row_ptr_B || !dcsr_col_ind_B || !dcsr_val_B || "
            "!dcsr_row_ptr_C_1 || !dcsr_row_ptr_C_2 || !d_alpha || !d_beta");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A.data(), sizeof(I) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind_A, hcsr_col_ind_A.data(), sizeof(J) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val_A, hcsr_val_A.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_B, hcsr_row_ptr_B.data(), sizeof(I) * (k + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind_B, hcsr_col_ind_B.data(), sizeof(J) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val_B, hcsr_val_B.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create matrices
    hipsparseSpMatDescr_t A, B, C1, C2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&A,
                                             m,
                                             k,
                                             nnz_A,
                                             dcsr_row_ptr_A,
                                             dcsr_col_ind_A,
                                             dcsr_val_A,
                                             typeI,
                                             typeJ,
                                             idxBaseA,
                                             typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&B,
                                             k,
                                             n,
                                             nnz_B,
                                             dcsr_row_ptr_B,
                                             dcsr_col_ind_B,
                                             dcsr_val_B,
                                             typeI,
                                             typeJ,
                                             idxBaseB,
                                             typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(
        &C1, m, n, 0, dcsr_row_ptr_C_1, nullptr, nullptr, typeI, typeJ, idxBaseC, typeT));
    // Query SpGEMM work estimation buffer
    size_t bufferSize1;
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_workEstimation(
        handle, transA, transB, A, B, C1, alg, descr, &bufferSize1, nullptr));


    auto externalBuffer1_managed
        = hipsparse_unique_ptr{device_malloc(bufferSize1), device_free};
    void* externalBuffer1 = (void*)externalBuffer1_managed.get();

    // SpGEMMreuse work estimation
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_workEstimation(
        handle, transA, transB, A, B, C1, alg, descr, &bufferSize1, externalBuffer1));
    // Query SpGEMM_nnz

    // SpGEMM work estimation
    size_t bufferSize2, bufferSize3, bufferSize4, bufferSize5;

    void * externalBuffer2 = nullptr, *externalBuffer3 = nullptr, *externalBuffer4 = nullptr,
         *externalBuffer5 = nullptr;

    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_nnz(handle,
                                                   transA,
                                                   transB,
                                                   A,
                                                   B,
                                                   C1,
                                                   alg,
                                                   descr,
                                                   &bufferSize2,
                                                   externalBuffer2,
                                                   &bufferSize3,
                                                   externalBuffer3,
                                                   &bufferSize4,
                                                   externalBuffer4));

    auto externalBuffer2_managed
      = hipsparse_unique_ptr{device_malloc(bufferSize2), device_free};
    externalBuffer2 = (void*)externalBuffer2_managed.get();
    auto externalBuffer3_managed
      = hipsparse_unique_ptr{device_malloc(bufferSize3), device_free};
    externalBuffer3 = (void*)externalBuffer3_managed.get();
    auto externalBuffer4_managed
      = hipsparse_unique_ptr{device_malloc(bufferSize4), device_free};
    externalBuffer4 = (void*)externalBuffer4_managed.get();

    CHECK_HIP_ERROR(hipMalloc(&externalBuffer2, bufferSize2));
    CHECK_HIP_ERROR(hipMalloc(&externalBuffer3, bufferSize3));
    CHECK_HIP_ERROR(hipMalloc(&externalBuffer4, bufferSize4));

    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_nnz(handle,
                                                   transA,
                                                   transB,
                                                   A,
                                                   B,
                                                   C1,
                                                   alg,
                                                   descr,
                                                   &bufferSize2,
                                                   externalBuffer2,
                                                   &bufferSize3,
                                                   externalBuffer3,
                                                   &bufferSize4,
                                                   externalBuffer4));

    // We can already free buffer1
    externalBuffer1_managed.reset(nullptr);
    externalBuffer1 = nullptr;

    externalBuffer2_managed.reset(nullptr);
    externalBuffer2 = nullptr;

    // Get nnz of C
    int64_t rows_C, cols_C, nnz_C_1, nnz_C_2;
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseSpMatGetSize(C1, &rows_C, &cols_C, &nnz_C_1));
    nnz_C_2 = nnz_C_1;

    // Allocate C
    auto dcsr_col_ind_C_1_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz_C_1), device_free};
    auto dcsr_val_C_1_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C_1), device_free};
    auto dcsr_col_ind_C_2_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz_C_2), device_free};
    auto dcsr_val_C_2_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C_2), device_free};

    J* dcsr_col_ind_C_1 = (J*)dcsr_col_ind_C_1_managed.get();
    T* dcsr_val_C_1     = (T*)dcsr_val_C_1_managed.get();
    J* dcsr_col_ind_C_2 = (J*)dcsr_col_ind_C_2_managed.get();
    T* dcsr_val_C_2     = (T*)dcsr_val_C_2_managed.get();
    if(!dcsr_col_ind_C_1 || !dcsr_val_C_1 || !dcsr_col_ind_C_2 || !dcsr_val_C_2)
    {
        verify_hipsparse_status_success(
            HIPSPARSE_STATUS_ALLOC_FAILED,
            "!dcsr_col_ind_C_1 || !dcsr_val_C_1 || !dcsr_col_ind_C_2 || !dcsr_val_C_2");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Set C pointers
    CHECK_HIPSPARSE_ERROR(
        hipsparseCsrSetPointers(C1, dcsr_row_ptr_C_1, dcsr_col_ind_C_1, dcsr_val_C_1));

    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_copy(
        handle, transA, transB, A, B, C1, alg, descr, &bufferSize5, externalBuffer5));

    auto externalBuffer5_managed
      = hipsparse_unique_ptr{device_malloc(bufferSize5), device_free};
    externalBuffer5 = (void*)externalBuffer5_managed.get();

    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_copy(
        handle, transA, transB, A, B, C1, alg, descr, &bufferSize5, externalBuffer5));

    externalBuffer3_managed.reset(nullptr);
    externalBuffer3 = nullptr;

    // Query SpGEMM compute buffer

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_compute(
        handle, transA, transB, &h_alpha, A, B, &h_beta, C1, typeT, alg, descr));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_C_2, dcsr_row_ptr_C_1, sizeof(I) * (m + 1), hipMemcpyDeviceToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind_C_2, dcsr_col_ind_C_1, sizeof(J) * nnz_C_1, hipMemcpyDeviceToDevice))
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val_C_2, dcsr_val_C_1, sizeof(T) * nnz_C_1, hipMemcpyDeviceToDevice));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&C2,
                                             m,
                                             n,
                                             nnz_C_1,
                                             dcsr_row_ptr_C_2,
                                             dcsr_col_ind_C_2,
                                             dcsr_val_C_2,
                                             typeI,
                                             typeJ,
                                             idxBaseC,
                                             typeT));

    CHECK_HIPSPARSE_ERROR(hipsparseSpGEMMreuse_compute(
        handle, transA, transB, d_alpha, A, B, d_beta, C1, typeT, alg, descr));

    externalBuffer4_managed.reset(nullptr);
    externalBuffer4 = nullptr;

    externalBuffer5_managed.reset(nullptr);
    externalBuffer5 = nullptr;

    // Copy output from device to CPU
    std::vector<I> hcsr_row_ptr_C_1(m + 1);
    std::vector<I> hcsr_row_ptr_C_2(m + 1);
    std::vector<J> hcsr_col_ind_C_1(nnz_C_1);
    std::vector<J> hcsr_col_ind_C_2(nnz_C_2);
    std::vector<T> hcsr_val_C_1(nnz_C_1);
    std::vector<T> hcsr_val_C_2(nnz_C_2);

    CHECK_HIP_ERROR(hipMemcpy(
        hcsr_row_ptr_C_1.data(), dcsr_row_ptr_C_1, sizeof(I) * (m + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        hcsr_row_ptr_C_2.data(), dcsr_row_ptr_C_2, sizeof(I) * (m + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        hcsr_col_ind_C_1.data(), dcsr_col_ind_C_1, sizeof(J) * nnz_C_1, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        hcsr_col_ind_C_2.data(), dcsr_col_ind_C_2, sizeof(J) * nnz_C_2, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hcsr_val_C_1.data(), dcsr_val_C_1, sizeof(T) * nnz_C_1, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hcsr_val_C_2.data(), dcsr_val_C_2, sizeof(T) * nnz_C_2, hipMemcpyDeviceToHost));

    // Compute SpGEMM nnz of C on host
    std::vector<I> hcsr_row_ptr_C_gold(m + 1);

    int64_t nnz_C_gold = csrgemm2_nnz(m,
                                      n,
                                      k,
                                      &h_alpha,
                                      hcsr_row_ptr_A.data(),
                                      hcsr_col_ind_A.data(),
                                      hcsr_row_ptr_B.data(),
                                      hcsr_col_ind_B.data(),
                                      (const T*)nullptr,
                                      (const I*)nullptr,
                                      (const J*)nullptr,
                                      hcsr_row_ptr_C_gold.data(),
                                      idxBaseA,
                                      idxBaseB,
                                      idxBaseC,
                                      HIPSPARSE_INDEX_BASE_ZERO);
    // Verify nnz and row pointer array
    unit_check_general(1, 1, 1, &nnz_C_gold, &nnz_C_1);
    unit_check_general(1, 1, 1, &nnz_C_gold, &nnz_C_2);
    unit_check_general(1, m + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C_1.data());
    unit_check_general(1, m + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C_2.data());

    // Compute SpGEMM on host
    std::vector<J> hcsr_col_ind_C_gold(nnz_C_gold);
    std::vector<T> hcsr_val_C_gold(nnz_C_gold);

    csrgemm2(m,
             n,
             k,
             &h_alpha,
             hcsr_row_ptr_A.data(),
             hcsr_col_ind_A.data(),
             hcsr_val_A.data(),
             hcsr_row_ptr_B.data(),
             hcsr_col_ind_B.data(),
             hcsr_val_B.data(),
             (const T*)nullptr,
             (const I*)nullptr,
             (const J*)nullptr,
             (const T*)nullptr,
             hcsr_row_ptr_C_gold.data(),
             hcsr_col_ind_C_gold.data(),
             hcsr_val_C_gold.data(),
             idxBaseA,
             idxBaseB,
             idxBaseC,
             HIPSPARSE_INDEX_BASE_ZERO);

    // Verify column and value array

    unit_check_general(1, nnz_C_gold, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C_1.data());

    unit_check_general(1, nnz_C_gold, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C_2.data());

    unit_check_general(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C_1.data());
    unit_check_general(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C_2.data());

    // Clean up
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(B));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(C1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(C2));

    return HIPSPARSE_STATUS_SUCCESS;
#endif
}

#endif // TESTING_SPGEMM_CSR_HPP
