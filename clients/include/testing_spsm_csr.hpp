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
#ifdef __HIP_PLATFORM_NVIDIA__
    // do not test for bad args
    return;
#endif

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
    hipsparseOrder_t     order     = HIPSPARSE_ORDER_COL;
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
            handle, transA, transB, &alpha, A, B, C, dataType, alg, nullptr, &bsize),
        "Error: descr is nullptr");
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
            handle, transA, transB, &alpha, A, B, C, dataType, alg, nullptr, dbuf),
        "Error: descr is nullptr");
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
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSM_solve(handle, transA, transB, &alpha, A, B, C, dataType, alg, nullptr, dbuf),
        "Error: descr is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseSpSM_destroyDescr(descr), "success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(A), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(B), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnMat(C), "success");
#endif
}

template <typename I, typename J, typename T>
hipsparseStatus_t testing_spsm_csr(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
    J                    m        = argus.M;
    J                    n        = argus.N;
    J                    k        = argus.K;
    T                    h_alpha  = make_DataType<T>(argus.alpha);
    hipsparseOperation_t transA   = argus.transA;
    hipsparseOperation_t transB   = argus.transB;
    hipsparseOrder_t     orderB   = argus.orderB;
    hipsparseOrder_t     orderC   = argus.orderC;
    hipsparseIndexBase_t idx_base = argus.idx_base;
    hipsparseDiagType_t  diag     = argus.diag_type;
    hipsparseFillMode_t  uplo     = argus.fill_mode;
    hipsparseSpSMAlg_t   alg      = HIPSPARSE_SPSM_ALG_DEFAULT;

    std::string filename = argus.filename;

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

    I nnz;
    if(!generate_csr_matrix(filename, 
                            m, 
                            n, 
                            nnz, 
                            hcsr_row_ptr, 
                            hcsr_col_ind, 
                            hcsr_val, 
                            idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // Some matrix properties
    J B_m = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : k;
    J B_n = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : m;
    J C_m = m;
    J C_n = k;

    int ld_multiplier_B = 1;
    int ld_multiplier_C = 1;

    int64_t ldb = (orderB == HIPSPARSE_ORDER_COL)
                      ? ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? (int64_t(ld_multiplier_B) * m)
                                                               : (int64_t(ld_multiplier_B) * k))
                      : ((transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? (int64_t(ld_multiplier_B) * k)
                                                               : (int64_t(ld_multiplier_B) * m));
    int64_t ldc = (orderC == HIPSPARSE_ORDER_COL) ? (int64_t(ld_multiplier_C) * m)
                                                      : (int64_t(ld_multiplier_C) * k);
    
    ldb = std::max(int64_t(1), ldb);
    ldc = std::max(int64_t(1), ldc);

    int64_t nrowB = (orderB == HIPSPARSE_ORDER_COL) ? ldb : B_m;
    int64_t ncolB = (orderB == HIPSPARSE_ORDER_COL) ? B_n : ldb;
    int64_t nrowC = (orderC == HIPSPARSE_ORDER_COL) ? ldc : C_m;
    int64_t ncolC = (orderC == HIPSPARSE_ORDER_COL) ? C_n : ldc;

    int64_t nnz_B = nrowB * ncolB;
    int64_t nnz_C = nrowC * ncolC;

    std::vector<T> hB(nnz_B);
    std::vector<T> hC_1(nnz_C);
    std::vector<T> hC_2(nnz_C);
    std::vector<T> hC_gold(nnz_C);

    hipsparseInit<T>(hB, 1, nnz_B);

    hC_1    = hB;
    hC_2    = hC_1;
    hC_gold = hC_1;

    // allocate memory on device
    auto dptr_managed    = hipsparse_unique_ptr{device_malloc(sizeof(I) * (m + 1)), device_free};
    auto dcol_managed    = hipsparse_unique_ptr{device_malloc(sizeof(J) * nnz), device_free};
    auto dval_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dB_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dC_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto dC_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dptr    = (I*)dptr_managed.get();
    J* dcol    = (J*)dcol_managed.get();
    T* dval    = (T*)dval_managed.get();
    T* dB      = (T*)dB_managed.get();
    T* dC_1    = (T*)dC_1_managed.get();
    T* dC_2    = (T*)dC_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(I) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(J) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    hipsparseSpSMDescr_t descr;
    CHECK_HIPSPARSE_ERROR(hipsparseSpSM_createDescr(&descr));

    // Create matrices
    hipsparseSpMatDescr_t A;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCsr(&A, m, m, nnz, dptr, dcol, dval, typeI, typeJ, idx_base, typeT));

    // Create dense matrices
    hipsparseDnMatDescr_t B, C1, C2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&B, B_m, B_n, ldb, dB, typeT, orderB));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C1, C_m, C_n, ldc, dC_1, typeT, orderC));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&C2, C_m, C_n, ldc, dC_2, typeT, orderC));

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
    CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC_1, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC_2, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

    J struct_pivot  = -1;
    J numeric_pivot = -1;
    host_csrsm(m,
                k,
                nnz,
                transA,
                transB,
                h_alpha,
                hcsr_row_ptr,
                hcsr_col_ind,
                hcsr_val,
                hB,
                (J)ldb,
                orderB,
                hC_gold,
                (J)ldc,
                orderC,
                diag,
                uplo,
                idx_base,
                &struct_pivot,
                &numeric_pivot);

    if(struct_pivot == -1 && numeric_pivot == -1)
    {
        unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_1.data());
        unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_2.data());
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
