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
#ifndef TESTING_SPSM_CSR_HPP
#define TESTING_SPSM_CSR_HPP

#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse_test;

void testing_spsm_csr_bad_arg(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
    int64_t              m         = 100;
    int64_t              n         = 100;
    int64_t              k         = 100;
    int64_t              nnz       = 100;
    int64_t              safe_size = 100;
    float                alpha     = 0.6;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idxBase   = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseOrder_t     order     = HIPSPARSE_ORDER_COLUMN;
    hipsparseIndexType_t idxType   = HIPSPARSE_INDEX_32I;
    hipDataType          dataType  = HIP_R_32F;
    hipsparseSpSMAlg_t   alg       = HIPSPARSE_SPSM_ALG_DEFAULT;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dC_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*   dptr = (int*)dptr_managed.get();
    int*   dcol = (int*)dcol_managed.get();
    float* dval = (float*)dval_managed.get();
    float* dB   = (float*)dB_managed.get();
    float* dC   = (float*)dC_managed.get();
    void*  dbuf = (void*)dbuf_managed.get();

    if(!dval || !dptr || !dcol || !dB || !dC || !dbuf)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // SpSM structures
    hipsparseSpMatDescr_t A;
    hipsparseDnMatDescr_t B, C;

    hipsparseSpSMDescr_t descr;

    verify_hipsparse_status_success(hipsparseSpSM_createDescr(&descr), "success");

    size_t bsize;

    // Create SpSM structures
    verify_hipsparse_status_success(
        hipsparseCreateCsr(&A, m, n, nnz, dptr, dcol, dval, idxType, idxType, idxBase, dataType),
        "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&B, m, k, m, dB, dataType, order),
                                    "success");
    verify_hipsparse_status_success(hipsparseCreateDnMat(&C, m, k, m, dC, dataType, order),
                                    "success");

    // SpSM buffer
    verify_hipsparse_status_invalid_handle(hipsparseSpSM_bufferSize(
        nullptr, transA, transB, &alpha, A, B, C, dataType, alg, descr, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_bufferSize(
            handle, transA, transB, nullptr, A, B, C, dataType, alg, descr, &bsize),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_bufferSize(
            handle, transA, transB, &alpha, nullptr, B, C, dataType, alg, descr, &bsize),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_bufferSize(
            handle, transA, transB, &alpha, A, nullptr, C, dataType, alg, descr, &bsize),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_bufferSize(
            handle, transA, transB, &alpha, A, B, nullptr, dataType, alg, descr, &bsize),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_bufferSize(
            handle, transA, transB, &alpha, A, B, C, dataType, alg, descr, nullptr),
        "Error: bsize is nullptr");

    // SpSM analysis
    verify_hipsparse_status_invalid_handle(hipsparseSpSM_analysis(
        nullptr, transA, transB, &alpha, A, B, C, dataType, alg, descr, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_analysis(
            handle, transA, transB, nullptr, A, B, C, dataType, alg, descr, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_analysis(
            handle, transA, transB, &alpha, nullptr, B, C, dataType, alg, descr, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_analysis(
            handle, transA, transB, &alpha, A, nullptr, C, dataType, alg, descr, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_analysis(
            handle, transA, transB, &alpha, A, B, nullptr, dataType, alg, descr, dbuf),
        "Error: C is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_analysis(
            handle, transA, transB, &alpha, A, B, C, dataType, alg, descr, nullptr),
        "Error: dbuf is nullptr");

    // SpSM solve
    verify_hipsparse_status_invalid_handle(
        hipsparseSpSM_solve(nullptr, transA, transB, &alpha, A, B, C, dataType, alg, descr, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_solve(handle, transA, transB, nullptr, A, B, C, dataType, alg, descr, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_solve(
            handle, transA, transB, &alpha, nullptr, B, C, dataType, alg, descr, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_solve(
            handle, transA, transB, &alpha, A, nullptr, C, dataType, alg, descr, dbuf),
        "Error: B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_solve(
            handle, transA, transB, &alpha, A, B, nullptr, dataType, alg, descr, dbuf),
        "Error: C is nullptr");
#if(!defined(CUDART_VERSION))
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_solve(handle, transA, transB, &alpha, A, B, C, dataType, alg, descr, nullptr),
        "Error: dbuf is nullptr");
#endif

    // Destruct
    verify_hipsparse_status_success(hipsparseSpSM_destroyDescr(descr), "success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(A), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(B), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(C), "success");
#endif
}

template <typename I, typename J, typename T>
hipsparseStatus_t testing_spsm_csr(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
    T                    h_alpha  = make_DataType<T>(2.3);
    hipsparseOperation_t transA   = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB   = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idx_base = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseDiagType_t  diag     = HIPSPARSE_DIAG_TYPE_NON_UNIT;
    hipsparseFillMode_t  uplo     = HIPSPARSE_FILL_MODE_LOWER;
    hipsparseOrder_t     order    = HIPSPARSE_ORDER_COLUMN;
    hipsparseSpSMAlg_t   alg      = HIPSPARSE_SPSM_ALG_DEFAULT;

    std::string filename = hipsparse_exepath() + "../matrices/nos3.bin";

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

    // hipSPARSE handle
    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    // Host structures
    std::vector<I> hcsr_row_ptr;
    std::vector<J> hcol_ind;
    std::vector<T> hval;

    // Initial Data on CPU
    srand(12345ULL);

    J m;
    J n;
    I nnz;
    J k = 16;

    if(read_bin_matrix(filename.c_str(), m, n, nnz, hcsr_row_ptr, hcol_ind, hval, idx_base) != 0)
    {
        fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    std::vector<T> hB(m * k);
    std::vector<T> hC_1(m * k);
    std::vector<T> hC_2(m * k);
    std::vector<T> hC_gold(m * k);

    hipsparseInit<T>(hB, 1, m * k);

    hC_1    = hB;
    hC_2    = hC_1;
    hC_gold = hC_1;

    // allocate memory on device
    auto dptr_managed    = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto dcol_managed    = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz), device_free};
    auto dval_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dB_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * k), device_free};
    auto dC_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * k), device_free};
    auto dC_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * k), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dptr    = (I*)dptr_managed.get();
    J* dcol    = (J*)dcol_managed.get();
    T* dval    = (T*)dval_managed.get();
    T* dB      = (T*)dB_managed.get();
    T* dC_1    = (T*)dC_1_managed.get();
    T* dC_2    = (T*)dC_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();

    if(!dval || !dptr || !dcol || !dB || !dC_1 || !dC_2 || !d_alpha)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dval || !dptr || !dcol || !dB || "
                                        "!dC_1 || !dC_2 || !d_alpha");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(I) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcol_ind.data(), sizeof(J) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hval.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * m * k, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1.data(), sizeof(T) * m * k, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2.data(), sizeof(T) * m * k, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    hipsparseSpSMDescr_t descr;
    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_createDescr(&descr));

    // Create matrices
    hipsparseSpMatDescr_t A;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCsr(&A, m, n, nnz, dptr, dcol, dval, typeI, typeJ, idx_base, typeT));

    // Create dense matrices
    hipsparseDnMatDescr_t B, C1, C2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&B, m, k, m, dB, typeT, order));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C1, m, k, m, dC_1, typeT, order));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C2, m, k, m, dC_2, typeT, order));

    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMatSetAttribute(A, HIPSPARSE_SPMAT_FILL_MODE, &uplo, sizeof(uplo)));

    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMatSetAttribute(A, HIPSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));

    // Query SpSM buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_bufferSize(
        handle, transA, transB, &h_alpha, A, B, C1, typeT, alg, descr, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // HIPSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_analysis(
        handle, transA, transB, &h_alpha, A, B, C1, typeT, alg, descr, buffer));

    // HIPSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_analysis(
        handle, transA, transB, d_alpha, A, B, C2, typeT, alg, descr, buffer));

    // HIPSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSM_solve(handle, transA, transB, &h_alpha, A, B, C1, typeT, alg, descr, buffer));

    // HIPSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSM_solve(handle, transA, transB, d_alpha, A, B, C2, typeT, alg, descr, buffer));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC_1, sizeof(T) * m * k, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC_2, sizeof(T) * m * k, hipMemcpyDeviceToHost));

    J struct_pivot  = -1;
    J numeric_pivot = -1;
    host_csrsm(m,
               k,
               nnz,
               transA,
               transB,
               h_alpha,
               hcsr_row_ptr,
               hcol_ind,
               hval,
               hC_gold,
               m,
               diag,
               uplo,
               idx_base,
               &struct_pivot,
               &numeric_pivot);

    if(struct_pivot == -1 && numeric_pivot == -1)
    {
        unit_check_near(1, m * k, 1, hC_gold.data(), hC_1.data());
        unit_check_near(1, m * k, 1, hC_gold.data(), hC_2.data());
    }

    CHECK_HIP_ERROR(hipFree(buffer));

    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_destroyDescr(descr));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(B));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(C2));
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_SPSM_CSR_HPP
