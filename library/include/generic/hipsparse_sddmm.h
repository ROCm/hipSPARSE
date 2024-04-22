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
*  \brief Description: Calculate the buffer size required for the sampled dense dense matrix multiplication
*
*  \details
*  \p hipsparseSDDMM_bufferSize computes the required user allocated buffer size needed when computing the 
*  sampled dense dense matrix multiplication
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_bufferSize(hipsparseHandle_t          handle,
                                            hipsparseOperation_t       opA,
                                            hipsparseOperation_t       opB,
                                            const void*                alpha,
                                            hipsparseConstDnMatDescr_t A,
                                            hipsparseConstDnMatDescr_t B,
                                            const void*                beta,
                                            hipsparseSpMatDescr_t      C,
                                            hipDataType                computeType,
                                            hipsparseSDDMMAlg_t        alg,
                                            size_t*                    bufferSize);
#elif(CUDART_VERSION >= 11022)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_bufferSize(hipsparseHandle_t           handle,
                                            hipsparseOperation_t        opA,
                                            hipsparseOperation_t        opB,
                                            const void*                 alpha,
                                            const hipsparseDnMatDescr_t A,
                                            const hipsparseDnMatDescr_t B,
                                            const void*                 beta,
                                            hipsparseSpMatDescr_t       C,
                                            hipDataType                 computeType,
                                            hipsparseSDDMMAlg_t         alg,
                                            size_t*                     bufferSize);
#endif

/*! \ingroup generic_module
*  \brief Description: Preprocess step of the sampled dense dense matrix multiplication.
*
*  \details
*  \p hipsparseSDDMM_preprocess performs the required preprocessing used when computing the 
*  sampled dense dense matrix multiplication
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_preprocess(hipsparseHandle_t          handle,
                                            hipsparseOperation_t       opA,
                                            hipsparseOperation_t       opB,
                                            const void*                alpha,
                                            hipsparseConstDnMatDescr_t A,
                                            hipsparseConstDnMatDescr_t B,
                                            const void*                beta,
                                            hipsparseSpMatDescr_t      C,
                                            hipDataType                computeType,
                                            hipsparseSDDMMAlg_t        alg,
                                            void*                      tempBuffer);
#elif(CUDART_VERSION >= 11022)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_preprocess(hipsparseHandle_t           handle,
                                            hipsparseOperation_t        opA,
                                            hipsparseOperation_t        opB,
                                            const void*                 alpha,
                                            const hipsparseDnMatDescr_t A,
                                            const hipsparseDnMatDescr_t B,
                                            const void*                 beta,
                                            hipsparseSpMatDescr_t       C,
                                            hipDataType                 computeType,
                                            hipsparseSDDMMAlg_t         alg,
                                            void*                       tempBuffer);
#endif

/*! \ingroup generic_module
*  \brief  Description: Sampled Dense-Dense Matrix Multiplication.
*
*  \details
*  \ref hipsparseSDDMM multiplies the scalar \f$\alpha\f$ with the dense
*  \f$m \times k\f$ matrix \f$A\f$, the dense \f$k \times n\f$ matrix \f$B\f$, filtered by the sparsity pattern of the \f$m \times n\f$ sparse matrix \f$C\f$ and
*  adds the result to \f$C\f$ scaled by
*  \f$\beta\f$. The final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  such that
*  \f[
*    C := \alpha ( opA(A) \cdot opB(B) ) \cdot spy(C) + \beta C,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if opA == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T,   & \text{if opA == HIPSPARSE_OPERATION_TRANSPOSE} \\
*    \end{array}
*    \right.
*  \f],
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if opB == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T,   & \text{if opB == HIPSPARSE_OPERATION_TRANSPOSE} \\
*    \end{array}
*    \right.
*  \f]
*   and
*  \f[
*    spy(C)_ij = \left\{
*    \begin{array}{ll}
*        1 \text{if i == j},   & 0 \text{if i != j} \\
*    \end{array}
*    \right.
*  \f]
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM(hipsparseHandle_t          handle,
                                 hipsparseOperation_t       opA,
                                 hipsparseOperation_t       opB,
                                 const void*                alpha,
                                 hipsparseConstDnMatDescr_t A,
                                 hipsparseConstDnMatDescr_t B,
                                 const void*                beta,
                                 hipsparseSpMatDescr_t      C,
                                 hipDataType                computeType,
                                 hipsparseSDDMMAlg_t        alg,
                                 void*                      tempBuffer);
#elif(CUDART_VERSION >= 11022)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM(hipsparseHandle_t           handle,
                                 hipsparseOperation_t        opA,
                                 hipsparseOperation_t        opB,
                                 const void*                 alpha,
                                 const hipsparseDnMatDescr_t A,
                                 const hipsparseDnMatDescr_t B,
                                 const void*                 beta,
                                 hipsparseSpMatDescr_t       C,
                                 hipDataType                 computeType,
                                 hipsparseSDDMMAlg_t         alg,
                                 void*                       tempBuffer);
#endif

#ifdef __cplusplus
}
#endif
