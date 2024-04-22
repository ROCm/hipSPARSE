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
*  \brief Description: Create sparse matrix sparse matrix product descriptor
*  \details
*  \p hipsparseSpGEMM_createDescr creates a sparse matrix sparse matrix product descriptor. It should be
*  destroyed at the end using hipsparseSpGEMM_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_createDescr(hipsparseSpGEMMDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \brief Description: Destroy sparse matrix sparse matrix product descriptor
*  \details
*  \p hipsparseSpGEMM_destroyDescr destroys a sparse matrix sparse matrix product descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_destroyDescr(hipsparseSpGEMMDescr_t descr);
#endif

/*! \ingroup generic_module
*  \brief Description: Work estimation step of the sparse matrix sparse matrix product C' = alpha * A * B + beta * C 
*  where C', A, B, C are sparse matrices and C' and C have the same sparsity pattern.
*
*  \details
*  \p hipsparseSpGEMM_workEstimation is called twice. We call it to compute the size of the first required user allocated
*  buffer. After this buffer size is determined, the user allocates it and calls \p hipsparseSpGEMM_workEstimation
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
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                  &alpha, matA, matB, &beta, matC,
*                                  computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                  spgemmDesc, &bufferSize1, NULL);
*    hipMalloc((void**) &dBuffer1, bufferSize1);
*
*    // Determine number of intermediate product when computing A * B
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                    &alpha, matA, matB, &beta, matC,
*                                    computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                    spgemmDesc, &bufferSize1, dBuffer1);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_workEstimation(hipsparseHandle_t          handle,
                                                 hipsparseOperation_t       opA,
                                                 hipsparseOperation_t       opB,
                                                 const void*                alpha,
                                                 hipsparseConstSpMatDescr_t matA,
                                                 hipsparseConstSpMatDescr_t matB,
                                                 const void*                beta,
                                                 hipsparseSpMatDescr_t      matC,
                                                 hipDataType                computeType,
                                                 hipsparseSpGEMMAlg_t       alg,
                                                 hipsparseSpGEMMDescr_t     spgemmDescr,
                                                 size_t*                    bufferSize1,
                                                 void*                      externalBuffer1);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_workEstimation(hipsparseHandle_t      handle,
                                                 hipsparseOperation_t   opA,
                                                 hipsparseOperation_t   opB,
                                                 const void*            alpha,
                                                 hipsparseSpMatDescr_t  matA,
                                                 hipsparseSpMatDescr_t  matB,
                                                 const void*            beta,
                                                 hipsparseSpMatDescr_t  matC,
                                                 hipDataType            computeType,
                                                 hipsparseSpGEMMAlg_t   alg,
                                                 hipsparseSpGEMMDescr_t spgemmDescr,
                                                 size_t*                bufferSize1,
                                                 void*                  externalBuffer1);
#endif

/*! \ingroup generic_module
*  \brief Description: Compute step of the sparse matrix sparse matrix product C' = alpha * A * B + beta * C 
*  where C', A, B, C are sparse matrices and C' and C have the same sparsity pattern.
*
*  \details
*  \p hipsparseSpGEMM_compute is called twice. First to compute the size of the second required user allocated
*  buffer. After this buffer size is determined, the user allocates it and calls \p hipsparseSpGEMM_compute
*  a second time with the newly allocated buffer passed in. This second call performs the actual computation 
*  of C' = alpha * A * B (the result is stored in the temporary buffers).
*  
*  \par Example (See full example below)
*  \code{.c}
*    void*  dBuffer2  = NULL; 
*    size_t bufferSize2 = 0;
*
*    size_t bufferSize2 = 0;
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                            spgemmDesc, &bufferSize2, NULL);
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*
*    // compute the intermediate product of A * B
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                            spgemmDesc, &bufferSize2, dBuffer2);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_compute(hipsparseHandle_t          handle,
                                          hipsparseOperation_t       opA,
                                          hipsparseOperation_t       opB,
                                          const void*                alpha,
                                          hipsparseConstSpMatDescr_t matA,
                                          hipsparseConstSpMatDescr_t matB,
                                          const void*                beta,
                                          hipsparseSpMatDescr_t      matC,
                                          hipDataType                computeType,
                                          hipsparseSpGEMMAlg_t       alg,
                                          hipsparseSpGEMMDescr_t     spgemmDescr,
                                          size_t*                    bufferSize2,
                                          void*                      externalBuffer2);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_compute(hipsparseHandle_t      handle,
                                          hipsparseOperation_t   opA,
                                          hipsparseOperation_t   opB,
                                          const void*            alpha,
                                          hipsparseSpMatDescr_t  matA,
                                          hipsparseSpMatDescr_t  matB,
                                          const void*            beta,
                                          hipsparseSpMatDescr_t  matC,
                                          hipDataType            computeType,
                                          hipsparseSpGEMMAlg_t   alg,
                                          hipsparseSpGEMMDescr_t spgemmDescr,
                                          size_t*                bufferSize2,
                                          void*                  externalBuffer2);
#endif

/*! \ingroup generic_module
*  \brief Description: Copy step of the sparse matrix sparse matrix product C' = alpha * A * B + beta * C 
*  where C', A, B, C are sparse matrices and C' and C have the same sparsity pattern.
*
*  \details
*  \p hipsparseSpGEMM_copy is called once to copy the results (that are currently stored in the temporary arrays) 
*  to the output sparse matrix. If beta != 0, then the beta * C portion of the computation: C' = alpha * A * B + beta * C
*  is handled. This is possible because C' and C must have the same sparsity pattern.
*
*  \note The two user allocated temporary buffers can only be freed after the call to \p hipsparseSpGEMM_copy
*  
*  \par Example (Full example)
*  \code{.c}
*    hipsparseHandle_t     handle = NULL;
*    hipsparseSpMatDescr_t matA, matB, matC;
*    void*  dBuffer1  = NULL; 
*    void*  dBuffer2  = NULL;
*    size_t bufferSize1 = 0;  
*    size_t bufferSize2 = 0;
*
*    hipsparseCreate(&handle);
*
*    // Create sparse matrix A in CSR format
*    hipsparseCreateCsr(&matA, m, k, nnzA,
*                                        dcsr_row_ptrA, dcsr_col_indA, dcsr_valA,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matB, k, n, nnzB,
*                                        dcsr_row_ptrB, dcsr_col_indB, dcsr_valB,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matC, m, n, 0,
*                                        dcsr_row_ptrC, NULL, NULL,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*
*    hipsparseSpGEMMDescr_t spgemmDesc;
*    hipsparseSpGEMM_createDescr(&spgemmDesc);
*
*    // Determine size of first user allocated buffer
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                        spgemmDesc, &bufferSize1, NULL);
*    hipMalloc((void**) &dBuffer1, bufferSize1);
*
*    // Inspect the matrices A and B to determine the number of intermediate product in 
*    // C = alpha * A * B
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                        spgemmDesc, &bufferSize1, dBuffer1);
*
*    // Determine size of second user allocated buffer
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                                &alpha, matA, matB, &beta, matC,
*                                computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                spgemmDesc, &bufferSize2, NULL);
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*
*    // Compute C = alpha * A * B and store result in temporary buffers
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                        spgemmDesc, &bufferSize2, dBuffer2);
*
*    // Get matrix C non-zero entries C_nnz1
*    int64_t C_num_rows1, C_num_cols1, C_nnz1;
*    hipsparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
*
*    // Allocate the CSR structures for the matrix C
*    hipMalloc((void**) &dcsr_col_indC, C_nnz1 * sizeof(int));
*    hipMalloc((void**) &dcsr_valC,  C_nnz1 * sizeof(float));
*
*    // Update matC with the new pointers
*    hipsparseCsrSetPointers(matC, dcsr_row_ptrC, dcsr_col_indC, dcsr_valC);
*
*    // Copy the final products to the matrix C
*    hipsparseSpGEMM_copy(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc);
*
*    // Destroy matrix descriptors and handles
*    hipsparseSpGEMM_destroyDescr(spgemmDesc);
*    hipsparseDestroySpMat(matA);
*    hipsparseDestroySpMat(matB);
*    hipsparseDestroySpMat(matC);
*    hipsparseDestroy(handle);
* 
*    // Free device memory
*    hipFree(dBuffer1);
*    hipFree(dBuffer2);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_copy(hipsparseHandle_t          handle,
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
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_copy(hipsparseHandle_t      handle,
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
