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
#ifndef TESTING_SPSV_COO_HPP
#define TESTING_SPSV_COO_HPP

#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse_test;

void testing_spsv_coo_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVIDIA__
    // do not test for bad args
    return;
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
    int64_t              m         = 100;
    int64_t              n         = 100;
    int64_t              nnz       = 100;
    int64_t              safe_size = 100;
    float                alpha     = 0.6;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idxBase   = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexType_t idxType   = HIPSPARSE_INDEX_32I;
    hipDataType          dataType  = HIP_R_32F;
    hipsparseSpSVAlg_t   alg       = HIPSPARSE_SPSV_ALG_DEFAULT;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto drow_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dy_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*   drow = (int*)drow_managed.get();
    int*   dcol = (int*)dcol_managed.get();
    float* dval = (float*)dval_managed.get();
    float* dx   = (float*)dx_managed.get();
    float* dy   = (float*)dy_managed.get();
    void*  dbuf = (void*)dbuf_managed.get();

    if(!dval || !drow || !dcol || !dx || !dy || !dbuf)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // SpSV structures
    hipsparseSpMatDescr_t A;
    hipsparseDnVecDescr_t x, y;

    hipsparseSpSVDescr_t descr;

    verify_hipsparse_status_success(hipsparseSpSV_createDescr(&descr), "success");

    size_t bsize;

    // Create SpSV structures
    verify_hipsparse_status_success(
        hipsparseCreateCoo(&A, m, n, nnz, drow, dcol, dval, idxType, idxBase, dataType), "success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&x, m, dx, dataType), "success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&y, m, dy, dataType), "success");

    // SpSV buffer
    verify_hipsparse_status_invalid_handle(
        hipsparseSpSV_bufferSize(nullptr, transA, &alpha, A, x, y, dataType, alg, descr, &bsize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_bufferSize(handle, transA, nullptr, A, x, y, dataType, alg, descr, &bsize),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_bufferSize(
            handle, transA, &alpha, nullptr, x, y, dataType, alg, descr, &bsize),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_bufferSize(
            handle, transA, &alpha, A, nullptr, y, dataType, alg, descr, &bsize),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_bufferSize(
            handle, transA, &alpha, A, x, nullptr, dataType, alg, descr, &bsize),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_bufferSize(handle, transA, &alpha, A, x, y, dataType, alg, descr, nullptr),
        "Error: bsize is nullptr");

    // SpSV analysis
    verify_hipsparse_status_invalid_handle(
        hipsparseSpSV_analysis(nullptr, transA, &alpha, A, x, y, dataType, alg, descr, dbuf));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_analysis(handle, transA, nullptr, A, x, y, dataType, alg, descr, dbuf),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_analysis(handle, transA, &alpha, nullptr, x, y, dataType, alg, descr, dbuf),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_analysis(handle, transA, &alpha, A, nullptr, y, dataType, alg, descr, dbuf),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_analysis(handle, transA, &alpha, A, x, nullptr, dataType, alg, descr, dbuf),
        "Error: y is nullptr");
#if(!defined(CUDART_VERSION))
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_analysis(handle, transA, &alpha, A, x, y, dataType, alg, descr, nullptr),
        "Error: dbuf is nullptr");
#endif

    // SpSV solve
    verify_hipsparse_status_invalid_handle(
        hipsparseSpSV_solve(nullptr, transA, &alpha, A, x, y, dataType, alg, descr));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_solve(handle, transA, nullptr, A, x, y, dataType, alg, descr),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_solve(handle, transA, &alpha, nullptr, x, y, dataType, alg, descr),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_solve(handle, transA, &alpha, A, nullptr, y, dataType, alg, descr),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_solve(handle, transA, &alpha, A, x, nullptr, dataType, alg, descr),
        "Error: y is nullptr");
#if(!defined(CUDART_VERSION))
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpSV_solve(handle, transA, &alpha, A, x, y, dataType, alg, nullptr),
        "Error: descr is nullptr");
#endif

    // Destruct
    verify_hipsparse_status_success(hipsparseSpSV_destroyDescr(descr), "success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(A), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(x), "success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(y), "success");
#endif
}

template <typename I, typename T>
hipsparseStatus_t testing_spsv_coo(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
    T                    h_alpha  = make_DataType<T>(2.3);
    hipsparseOperation_t transA   = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idx_base = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseDiagType_t  diag     = HIPSPARSE_DIAG_TYPE_NON_UNIT;
    hipsparseFillMode_t  uplo     = HIPSPARSE_FILL_MODE_LOWER;
    hipsparseSpSVAlg_t   alg      = HIPSPARSE_SPSV_ALG_DEFAULT;

    std::string filename = hipsparse_exepath() + "../matrices/nos3.bin";

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
    I n;
    I nnz;

    if(read_bin_matrix(filename.c_str(), m, n, nnz, hrow_ptr, hcol_ind, hval, idx_base) != 0)
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

    std::vector<T> hx(m);
    std::vector<T> hy_1(m);
    std::vector<T> hy_2(m);
    std::vector<T> hy_gold(m);

    hipsparseInit<T>(hx, 1, m);
    hipsparseInit<T>(hy_1, 1, m);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    auto drow_managed    = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dcol_managed    = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dval_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dx_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* drow    = (I*)drow_managed.get();
    I* dcol    = (I*)dcol_managed.get();
    T* dval    = (T*)dval_managed.get();
    T* dx      = (T*)dx_managed.get();
    T* dy_1    = (T*)dy_1_managed.get();
    T* dy_2    = (T*)dy_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();

    if(!dval || !drow || !dcol || !dx || !dy_1 || !dy_2 || !d_alpha)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dval || !drow || !dcol || !dx || "
                                        "!dy_1 || !dy_2 || !d_alpha");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(drow, hrow_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcol_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hval.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    hipsparseSpSVDescr_t descr;
    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_createDescr(&descr));

    // Create matrices
    hipsparseSpMatDescr_t A;
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateCoo(&A, m, n, nnz, drow, dcol, dval, typeI, idx_base, typeT));

    // Create dense vectors
    hipsparseDnVecDescr_t x, y1, y2;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&x, m, dx, typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y1, m, dy_1, typeT));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y2, m, dy_2, typeT));

    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMatSetAttribute(A, HIPSPARSE_SPMAT_FILL_MODE, &uplo, sizeof(uplo)));

    CHECK_HIPSPARSE_ERROR(
        hipsparseSpMatSetAttribute(A, HIPSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));

    // Query SpSV buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_bufferSize(
        handle, transA, &h_alpha, A, x, y1, typeT, alg, descr, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // HIPSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSV_analysis(handle, transA, &h_alpha, A, x, y1, typeT, alg, descr, buffer));

    // HIPSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSV_analysis(handle, transA, d_alpha, A, x, y2, typeT, alg, descr, buffer));

    // HIPSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSV_solve(handle, transA, &h_alpha, A, x, y1, typeT, alg, descr));

    // HIPSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(
        hipsparseSpSV_solve(handle, transA, d_alpha, A, x, y2, typeT, alg, descr));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * m, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * m, hipMemcpyDeviceToHost));

    I struct_pivot  = -1;
    I numeric_pivot = -1;
    host_coosv(transA,
               m,
               nnz,
               h_alpha,
               hrow_ind,
               hcol_ind,
               hval,
               hx,
               hy_gold,
               diag,
               uplo,
               idx_base,
               &struct_pivot,
               &numeric_pivot);

    if(struct_pivot == -1 && numeric_pivot == -1)
    {
        unit_check_near(1, m, 1, hy_gold.data(), hy_1.data());
        unit_check_near(1, m, 1, hy_gold.data(), hy_2.data());
    }

    CHECK_HIP_ERROR(hipFree(buffer));

    CHECK_HIPSPARSE_ERROR(hipsparseSpSV_destroyDescr(descr));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(A));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(x));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y2));
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_SPSV_COO_HPP
