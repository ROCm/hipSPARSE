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
*  \brief Description: Create sparse matrix triangular solve with multiple rhs descriptor
*  \details
*  \p hipsparseSpSM_createDescr creates a sparse matrix triangular solve with multiple rhs descriptor. It should be
*  destroyed at the end using hipsparseSpSM_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_createDescr(hipsparseSpSMDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \brief Description: Destroy sparse matrix triangular solve with multiple rhs descriptor
*  \details
*  \p hipsparseSpSM_destroyDescr destroys a sparse matrix triangular solve with multiple rhs descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_destroyDescr(hipsparseSpSMDescr_t descr);
#endif

/*! \ingroup generic_module
*  \brief Description: Buffer size step of solution of triangular linear system op(A) * C = alpha * op(B),
*  where A is a sparse matrix in CSR storage format, B and C are dense matrices.
*
*  \details
*  \p hipsparseSpSV_bufferSize computes the required user allocated buffer size needed when computing the 
*  solution of triangular linear system op(A) * C = alpha * op(B), where A is a sparse matrix in CSR storage 
*  format, B and C are dense matrices.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpSMAlg_t          alg,
                                           hipsparseSpSMDescr_t        spsmDescr,
                                           size_t*                     bufferSize);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpSMAlg_t          alg,
                                           hipsparseSpSMDescr_t        spsmDescr,
                                           size_t*                     bufferSize);
#endif

/*! \ingroup generic_module
*  \brief Description: Analysis step of solution of triangular linear system op(A) * C = alpha * op(B),
*  where A is a sparse matrix in CSR storage format, B and C are dense vectors.
*
*  \details
*  \p hipsparseSpSV_analysis performs the required analysis used when computing the 
*  solution of triangular linear system op(A) * C = alpha * op(B),
*  where A is a sparse matrix in CSR storage format, B and C are dense vectors.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         hipsparseOperation_t        opB,
                                         const void*                 alpha,
                                         hipsparseConstSpMatDescr_t  matA,
                                         hipsparseConstDnMatDescr_t  matB,
                                         const hipsparseDnMatDescr_t matC,
                                         hipDataType                 computeType,
                                         hipsparseSpSMAlg_t          alg,
                                         hipsparseSpSMDescr_t        spsmDescr,
                                         void*                       externalBuffer);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         hipsparseOperation_t        opB,
                                         const void*                 alpha,
                                         const hipsparseSpMatDescr_t matA,
                                         const hipsparseDnMatDescr_t matB,
                                         const hipsparseDnMatDescr_t matC,
                                         hipDataType                 computeType,
                                         hipsparseSpSMAlg_t          alg,
                                         hipsparseSpSMDescr_t        spsmDescr,
                                         void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Sparse triangular system solve
*
*  \details
*  \p hipsparseSpSM_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR or COO storage format, a dense solution matrix
*  \f$C\f$ and the right-hand side \f$B\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot C = \alpha \cdot op(B),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      hipsparseOperation_t        opB,
                                      const void*                 alpha,
                                      hipsparseConstSpMatDescr_t  matA,
                                      hipsparseConstDnMatDescr_t  matB,
                                      const hipsparseDnMatDescr_t matC,
                                      hipDataType                 computeType,
                                      hipsparseSpSMAlg_t          alg,
                                      hipsparseSpSMDescr_t        spsmDescr,
                                      void*                       externalBuffer);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      hipsparseOperation_t        opB,
                                      const void*                 alpha,
                                      const hipsparseSpMatDescr_t matA,
                                      const hipsparseDnMatDescr_t matB,
                                      const hipsparseDnMatDescr_t matC,
                                      hipDataType                 computeType,
                                      hipsparseSpSMAlg_t          alg,
                                      hipsparseSpSMDescr_t        spsmDescr,
                                      void*                       externalBuffer);
#endif

#ifdef __cplusplus
}
#endif