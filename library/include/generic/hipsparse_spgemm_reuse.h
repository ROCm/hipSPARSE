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
*  \brief Description: Work estimation step of the sparse matrix sparse matrix product C' = alpha * A * B + beta * C 
*  where C', A, B, C are sparse matrices and C' and C have the same sparsity pattern.
*
*  \details
*  \p hipsparseSpGEMMreuse_workEstimation is called twice. We call it to compute the size of the first required user allocated
*  buffer. After this buffer size is determined, the user allocates it and calls \p hipsparseSpGEMMreuse_workEstimation
*  a second time with the newly allocated buffer passed in. This second call inspects the matrices A and B to 
*  determine the number of intermediate products that will result from multipltying A and B together.
*  
*  \par Example (See full example below)
*  \code{.c}
*    void*  dBuffer1  = NULL; 
*    size_t bufferSize1 = 0;
*
*    hipsparseSpGEMMDescr_t spgemmDesc;
*    hipsparseSpGEMM_createDescr(&spgemmDesc);
*
*    size_t bufferSize1 = 0;
*    hipsparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
*                                        HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc, 
*                                        &bufferSize1, NULL);
*    hipMalloc((void**) &dBuffer1, bufferSize1);
*
*    // Determine number of intermediate product when computing A * B
*    hipsparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
*                                        HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc, 
*                                        &bufferSize1, dBuffer1);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_workEstimation(hipsparseHandle_t          handle,
                                                      hipsparseOperation_t       opA,
                                                      hipsparseOperation_t       opB,
                                                      hipsparseConstSpMatDescr_t matA,
                                                      hipsparseConstSpMatDescr_t matB,
                                                      hipsparseSpMatDescr_t      matC,
                                                      hipsparseSpGEMMAlg_t       alg,
                                                      hipsparseSpGEMMDescr_t     spgemmDescr,
                                                      size_t*                    bufferSize1,
                                                      void*                      externalBuffer1);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_workEstimation(hipsparseHandle_t      handle,
                                                      hipsparseOperation_t   opA,
                                                      hipsparseOperation_t   opB,
                                                      hipsparseSpMatDescr_t  matA,
                                                      hipsparseSpMatDescr_t  matB,
                                                      hipsparseSpMatDescr_t  matC,
                                                      hipsparseSpGEMMAlg_t   alg,
                                                      hipsparseSpGEMMDescr_t spgemmDescr,
                                                      size_t*                bufferSize1,
                                                      void*                  externalBuffer1);
#endif

/*! \ingroup generic_module
*  \brief Description: Nnz calculation step of the sparse matrix sparse matrix product C' = alpha * A * B + beta * C 
*  where C', A, B, C are sparse matrices and C' and C have the same sparsity pattern.
*
*  \par Example (See full example below)
*  \code{.c}
*    // Determine size of second, third, and fourth user allocated buffer
*    hipsparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
*                                matC, HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                &bufferSize2, NULL, &bufferSize3, NULL,
*                                &bufferSize4, NULL);
*
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*    hipMalloc((void**) &dBuffer3, bufferSize3);
*    hipMalloc((void**) &dBuffer4, bufferSize4);
*
*    // COmpute sparsity pattern of C matrix and store in temporary buffers
*    hipsparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
*                                matC, HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                &bufferSize2, dBuffer2, &bufferSize3, dBuffer3,
*                                &bufferSize4, dBuffer4);
*
*    // We can now free buffer 1 and 2
*    hipFree(dBuffer1);
*    hipFree(dBuffer2);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_nnz(hipsparseHandle_t          handle,
                                           hipsparseOperation_t       opA,
                                           hipsparseOperation_t       opB,
                                           hipsparseConstSpMatDescr_t matA,
                                           hipsparseConstSpMatDescr_t matB,
                                           hipsparseSpMatDescr_t      matC,
                                           hipsparseSpGEMMAlg_t       alg,
                                           hipsparseSpGEMMDescr_t     spgemmDescr,
                                           size_t*                    bufferSize2,
                                           void*                      externalBuffer2,
                                           size_t*                    bufferSize3,
                                           void*                      externalBuffer3,
                                           size_t*                    bufferSize4,
                                           void*                      externalBuffer4);

#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_nnz(hipsparseHandle_t      handle,
                                           hipsparseOperation_t   opA,
                                           hipsparseOperation_t   opB,
                                           hipsparseSpMatDescr_t  matA,
                                           hipsparseSpMatDescr_t  matB,
                                           hipsparseSpMatDescr_t  matC,
                                           hipsparseSpGEMMAlg_t   alg,
                                           hipsparseSpGEMMDescr_t spgemmDescr,
                                           size_t*                bufferSize2,
                                           void*                  externalBuffer2,
                                           size_t*                bufferSize3,
                                           void*                  externalBuffer3,
                                           size_t*                bufferSize4,
                                           void*                  externalBuffer4);

#endif

/*! \ingroup generic_module
*  \brief Description: Copy step of the sparse matrix sparse matrix product C' = alpha * A * B + beta * C 
*  where C', A, B, C are sparse matrices and C' and C have the same sparsity pattern.
*
*  \par Example (See full example below)
*  \code{.c}
*    // Get matrix C non-zero entries nnzC
*    int64_t rowsC, colsC, nnzC;
*    hipsparseSpMatGetSize(matC, &rowsC, &colsC, &nnzC);
*
*    // Allocate matrix C
*    hipMalloc((void**) &dcsr_col_indC, sizeof(int) * nnzC);
*    hipMalloc((void**) &dcsr_valC,  sizeof(float) * nnzC);
*    
*    // Update matC with the new pointers. The C values array can be filled with data here
*    // which is used if beta != 0.
*    hipsparseCsrSetPointers(matC, dcsr_row_ptrC, dcsr_col_indC, dcsr_valC);
*
*    // Determine size of fifth user allocated buffer
*    hipsparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
*                                 HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                 &bufferSize5, NULL);
*
*    hipMalloc((void**) &dBuffer5, bufferSize5);
*    
*    // Copy data from temporary buffers to the newly allocated C matrix
*    hipsparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
*                                 HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                 &bufferSize5, dBuffer5);
*    
*    // We can now free buffer 3
*    hipFree(dBuffer3);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_copy(hipsparseHandle_t          handle,
                                            hipsparseOperation_t       opA,
                                            hipsparseOperation_t       opB,
                                            hipsparseConstSpMatDescr_t matA,
                                            hipsparseConstSpMatDescr_t matB,
                                            hipsparseSpMatDescr_t      matC,
                                            hipsparseSpGEMMAlg_t       alg,
                                            hipsparseSpGEMMDescr_t     spgemmDescr,
                                            size_t*                    bufferSize5,
                                            void*                      externalBuffer5);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_copy(hipsparseHandle_t      handle,
                                            hipsparseOperation_t   opA,
                                            hipsparseOperation_t   opB,
                                            hipsparseSpMatDescr_t  matA,
                                            hipsparseSpMatDescr_t  matB,
                                            hipsparseSpMatDescr_t  matC,
                                            hipsparseSpGEMMAlg_t   alg,
                                            hipsparseSpGEMMDescr_t spgemmDescr,
                                            size_t*                bufferSize5,
                                            void*                  externalBuffer5);
#endif

/*! \ingroup generic_module
*  \brief Description: Compute step of the sparse matrix sparse matrix product.
*
*  \par Full example
*  \code{.c}
*    int m = 2;
*    int k = 2;
*    int n = 3;
*    int nnzA = 4;
*    int nnzB = 4;
*    
*    float alpha{1.0f};
*    float beta{0.0f};
*
*    hipsparseOperation_t opA        = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipsparseOperation_t opB        = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipDataType computeType         = HIP_R_32F;
*
*    // A, B, and C are m×k, k×n, and m×n
*
*    // A
*    std::vector<int> hcsr_row_ptrA = {0, 2, 4};
*    std::vector<int> hcsr_col_indA = {0, 1, 0, 1};
*    std::vector<float> hcsr_valA = {1.0f, 2.0f, 3.0f, 4.0f};
*
*    // B
*    std::vector<int> hcsr_row_ptrB = {0, 2, 4};
*    std::vector<int> hcsr_col_indB = {1, 2, 0, 2};
*    std::vector<float> hcsr_valB = {5.0f , 6.0f, 7.0f, 8.0f};
*
*    // Device memory management: Allocate and copy A, B
*    int* dcsr_row_ptrA;
*    int* dcsr_col_indA;
*    float* dcsr_valA;
*    int* dcsr_row_ptrB;
*    int* dcsr_col_indB;
*    float* dcsr_valB;
*    int* dcsr_row_ptrC;
*    int* dcsr_col_indC;
*    float* dcsr_valC;
*    hipMalloc((void**)&dcsr_row_ptrA, (m + 1) * sizeof(int));
*    hipMalloc((void**)&dcsr_col_indA, nnzA * sizeof(int));
*    hipMalloc((void**)&dcsr_valA, nnzA * sizeof(float));
*    hipMalloc((void**)&dcsr_row_ptrB, (k + 1) * sizeof(int));
*    hipMalloc((void**)&dcsr_col_indB, nnzB * sizeof(int));
*    hipMalloc((void**)&dcsr_valB, nnzB * sizeof(float));
*    hipMalloc((void**)&dcsr_row_ptrC, (m + 1) * sizeof(int));
*
*    hipMemcpy(dcsr_row_ptrA, hcsr_row_ptrA.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsr_col_indA, hcsr_col_indA.data(), nnzA * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsr_valA, hcsr_valA.data(), nnzA * sizeof(float), hipMemcpyHostToDevice);
*    hipMemcpy(dcsr_row_ptrB, hcsr_row_ptrB.data(), (k + 1) * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsr_col_indB, hcsr_col_indB.data(), nnzB * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsr_valB, hcsr_valB.data(), nnzB * sizeof(float), hipMemcpyHostToDevice);
*
*    hipsparseHandle_t     handle = NULL;
*    hipsparseSpMatDescr_t matA, matB, matC;
*    void*  dBuffer1  = NULL; 
*    void*  dBuffer2  = NULL;
*    void*  dBuffer3  = NULL;
*    void*  dBuffer4  = NULL;
*    void*  dBuffer5  = NULL;
*    size_t bufferSize1 = 0;  
*    size_t bufferSize2 = 0;
*    size_t bufferSize3 = 0;
*    size_t bufferSize4 = 0;
*    size_t bufferSize5 = 0;
*
*    hipsparseCreate(&handle);
*
*    // Create sparse matrix A in CSR format
*    hipsparseCreateCsr(&matA, m, k, nnzA,
*                        dcsr_row_ptrA, dcsr_col_indA, dcsr_valA,
*                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matB, k, n, nnzB,
*                        dcsr_row_ptrB, dcsr_col_indB, dcsr_valB,
*                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matC, m, n, 0,
*                        dcsr_row_ptrC, NULL, NULL,
*                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*
*    hipsparseSpGEMMDescr_t spgemmDesc;
*    hipsparseSpGEMM_createDescr(&spgemmDesc);
*
*    // Determine size of first user allocated buffer
*    hipsparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
*                                        HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc, 
*                                        &bufferSize1, NULL);
*
*    hipMalloc((void**) &dBuffer1, bufferSize1);
*
*    // Inspect the matrices A and B to determine the number of intermediate product in 
*    // C = alpha * A * B
*    hipsparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
*                                        HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc, 
*                                        &bufferSize1, dBuffer1);
*
*    // Determine size of second, third, and fourth user allocated buffer
*    hipsparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
*                                matC, HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                &bufferSize2, NULL, &bufferSize3, NULL,
*                                &bufferSize4, NULL);
*
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*    hipMalloc((void**) &dBuffer3, bufferSize3);
*    hipMalloc((void**) &dBuffer4, bufferSize4);
*
*    // COmpute sparsity pattern of C matrix and store in temporary buffers
*    hipsparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
*                                matC, HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                &bufferSize2, dBuffer2, &bufferSize3, dBuffer3,
*                                &bufferSize4, dBuffer4);
*    
*    // We can now free buffer 1 and 2
*    hipFree(dBuffer1);
*    hipFree(dBuffer2);
*
*    // Get matrix C non-zero entries nnzC
*    int64_t rowsC, colsC, nnzC;
*    hipsparseSpMatGetSize(matC, &rowsC, &colsC, &nnzC);
*
*    // Allocate matrix C
*    hipMalloc((void**) &dcsr_col_indC, sizeof(int) * nnzC);
*    hipMalloc((void**) &dcsr_valC,  sizeof(float) * nnzC);
*    
*    // Update matC with the new pointers. The C values array can be filled with data here
*    // which is used if beta != 0.
*    hipsparseCsrSetPointers(matC, dcsr_row_ptrC, dcsr_col_indC, dcsr_valC);
*
*    // Determine size of fifth user allocated buffer
*    hipsparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
*                                 HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                 &bufferSize5, NULL);
*
*    hipMalloc((void**) &dBuffer5, bufferSize5);
*    
*    // Copy data from temporary buffers to the newly allocated C matrix
*    hipsparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
*                                 HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                 &bufferSize5, dBuffer5);
*    
*    // We can now free buffer 3
*    hipFree(dBuffer3);
*
*    // Compute C' = alpha * A * B + beta * C
*    hipsparseSpGEMMreuse_compute(handle, opA, opB, &alpha, matA, matB, &beta,
*                                    matC, computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                    spgemmDesc);
*
*    // Copy results back to host if required using hipsparseCsrGet...
*
*    // Update dcsr_valA, dcsr_valB with new values
*    for(size_t i = 0; i < hcsr_valA.size(); i++){ hcsr_valA[i] = 1.0f; }
*    for(size_t i = 0; i < hcsr_valB.size(); i++){ hcsr_valB[i] = 2.0f; }
*
*    hipMemcpy(dcsr_valA, hcsr_valA.data(), sizeof(float) * nnzA, hipMemcpyHostToDevice);
*    hipMemcpy(dcsr_valB, hcsr_valB.data(), sizeof(float) * nnzB, hipMemcpyHostToDevice);
*    
*    // Compute C' = alpha * A * B + beta * C again with the new A and B values
*    hipsparseSpGEMMreuse_compute(handle, opA, opB, &alpha, matA, matB, &beta,
*                                    matC, computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                    spgemmDesc);
*
*    // Copy results back to host if required using hipsparseCsrGet...
*
*    // Destroy matrix descriptors and handles
*    hipsparseSpGEMM_destroyDescr(spgemmDesc);
*    hipsparseDestroySpMat(matA);
*    hipsparseDestroySpMat(matB);
*    hipsparseDestroySpMat(matC);
*    hipsparseDestroy(handle);
*
*    // Free device memory
*    hipFree(dBuffer4);
*    hipFree(dBuffer5);
*    hipFree(dcsr_row_ptrA);
*    hipFree(dcsr_col_indA);
*    hipFree(dcsr_valA);
*    hipFree(dcsr_row_ptrB);
*    hipFree(dcsr_col_indB);
*    hipFree(dcsr_valB);
*    hipFree(dcsr_row_ptrC);
*    hipFree(dcsr_col_indC);
*    hipFree(dcsr_valC);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_compute(hipsparseHandle_t          handle,
                                               hipsparseOperation_t       opA,
                                               hipsparseOperation_t       opB,
                                               const void*                alpha,
                                               hipsparseConstSpMatDescr_t matA,
                                               hipsparseConstSpMatDescr_t matB,
                                               const void*                beta,
                                               hipsparseSpMatDescr_t      matC,
                                               hipDataType                computeType,
                                               hipsparseSpGEMMAlg_t       alg,
                                               hipsparseSpGEMMDescr_t     spgemmDescr);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_compute(hipsparseHandle_t      handle,
                                               hipsparseOperation_t   opA,
                                               hipsparseOperation_t   opB,
                                               const void*            alpha,
                                               hipsparseSpMatDescr_t  matA,
                                               hipsparseSpMatDescr_t  matB,
                                               const void*            beta,
                                               hipsparseSpMatDescr_t  matC,
                                               hipDataType            computeType,
                                               hipsparseSpGEMMAlg_t   alg,
                                               hipsparseSpGEMMDescr_t spgemmDescr);
#endif

#ifdef __cplusplus
}
#endif