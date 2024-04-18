/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Description: Calculate the buffer size required for the sparse matrix multiplication with a dense matrix
*
*  \details
*  \p hipsparseSpMM_bufferSize computes the required user allocated buffer size needed when computing the 
*  sparse matrix multiplication with a dense matrix
*
*  See full example below
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           size_t*                     bufferSize);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           size_t*                     bufferSize);
#endif

/*! \ingroup generic_module
*  \brief Description: Preprocess step of the sparse matrix multiplication with a dense matrix.
*
*  \details
*  \p hipsparseSpMM_preprocess performs the required preprocessing used when computing the 
*  sparse matrix multiplication with a dense matrix
*
*  See full example below
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           void*                       externalBuffer);
#elif(CUDART_VERSION >= 11021)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Compute the sparse matrix multiplication with a dense matrix
*
*  \details
*  \p hipsparseSpMM computes sparse matrix multiplication with a dense matrix
*
*  \par Example
*  \code{.c}
*    // A, B, and C are m×k, k×n, and m×n
*    int m = 3, n = 5, k = 4;
*    int ldb = n, ldc = n;
*    int nnz_A = 8, nnz_B = 20, nnz_C = 15;
*    hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipsparseOperation_t transC = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipsparseOrder_t order = HIPSPARSE_ORDER_ROW;
*
*    // alpha and beta
*    float alpha = 0.5f;
*    float beta  = 0.25f;
*
*    std::vector<int> hcsr_row_ptr = {0, 3, 5, 8};
*    std::vector<int> hcsr_col_ind = {0, 1, 3, 1, 2, 0, 2, 3}; 
*    std::vector<float> hcsr_val     = {1, 2, 3, 4, 5, 6, 7, 8}; 
*
*    std::vector<float> hB(nnz_B, 1.0f);
*    std::vector<float> hC(nnz_C, 1.0f);
*
*    int *dcsr_row_ptr;
*    int *dcsr_col_ind;
*    float *dcsr_val;
*    hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz_A);
*    hipMalloc((void**)&dcsr_val, sizeof(float) * nnz_A);
*
*    hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice);
*    hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz_A, hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseSpMatDescr_t matA;
*    hipsparseCreateCsr(&matA, m, k, nnz_A,
*                        dcsr_row_ptr, dcsr_col_ind, dcsr_val,
*                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*
*    // Allocate memory for the matrix B
*    float* dB;
*    hipMalloc((void**)&dB, sizeof(float) * nnz_B);
*    hipMemcpy(dB, hB.data(), sizeof(float) * nnz_B, hipMemcpyHostToDevice);
*
*    hipsparseDnMatDescr_t matB;
*    hipsparseCreateDnMat(&matB, k, n, ldb, dB, HIP_R_32F, order);
*
*    // Allocate memory for the resulting matrix C
*    float* dC;
*    hipMalloc((void**)&dC, sizeof(float) * nnz_C);
*    hipMemcpy(dC, hC.data(), sizeof(float) * nnz_C, hipMemcpyHostToDevice);
*
*    hipsparseDnMatDescr_t matC;
*    hipsparseCreateDnMat(&matC, m, n, ldc, dC, HIP_R_32F, HIPSPARSE_ORDER_ROW);
*
*    // Compute buffersize
*    size_t bufferSize;
*    hipsparseSpMM_bufferSize(handle,
*                             transA,
*                             transB,
*                             &alpha,
*                             matA,
*                             matB,
*                             &beta,
*                             matC,
*                             HIP_R_32F,
*                             HIPSPARSE_MM_ALG_DEFAULT,
*                             &bufferSize);
*
*    void* buffer;
*    hipMalloc(&buffer, bufferSize);
*
*    // Preprocess operation (Optional)
*    hipsparseSpMM_preprocess(handle,
*                            transA,
*                            transB,
*                            &alpha,
*                            matA,
*                            matB,
*                            &beta,
*                            matC,
*                            HIP_R_32F,
*                            HIPSPARSE_MM_ALG_DEFAULT,
*                            &buffer);
*
*    // Perform operation
*    hipsparseSpMM(handle,
*                 transA,
*                 transB,
*                 &alpha,
*                 matA,
*                 matB,
*                 &beta,
*                 matC,
*                 HIP_R_32F,
*                 HIPSPARSE_MM_ALG_DEFAULT,
*                 &buffer);
*
*    // Copy device to host
*    hipMemcpy(hC.data(), dC, sizeof(float) * nnz_C, hipMemcpyDeviceToHost);
*
*    // Destroy matrix descriptors and handles
*    hipsparseDestroySpMat(matA);
*    hipsparseDestroyDnMat(matB);
*    hipsparseDestroyDnMat(matC);
*    hipsparseDestroy(handle);
*
*    hipFree(buffer);
*    hipFree(dcsr_row_ptr);
*    hipFree(dcsr_col_ind);
*    hipFree(dcsr_val);
*    hipFree(dB);
*    hipFree(dC);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                hipsparseOperation_t        opB,
                                const void*                 alpha,
                                hipsparseConstSpMatDescr_t  matA,
                                hipsparseConstDnMatDescr_t  matB,
                                const void*                 beta,
                                const hipsparseDnMatDescr_t matC,
                                hipDataType                 computeType,
                                hipsparseSpMMAlg_t          alg,
                                void*                       externalBuffer);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                hipsparseOperation_t        opB,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnMatDescr_t matB,
                                const void*                 beta,
                                const hipsparseDnMatDescr_t matC,
                                hipDataType                 computeType,
                                hipsparseSpMMAlg_t          alg,
                                void*                       externalBuffer);
#endif

#ifdef __cplusplus
}
#endif