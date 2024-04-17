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

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level3_module
*  \brief Dense matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXgemmi multiplies the scalar \f$\alpha\f$ with a dense \f$m \times k\f$
*  matrix \f$A\f$ and the sparse \f$k \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
*  is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
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
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const float*      alpha,
                                  const float*      A,
                                  int               lda,
                                  const float*      cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const float*      beta,
                                  float*            C,
                                  int               ldc);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const double*     alpha,
                                  const double*     A,
                                  int               lda,
                                  const double*     cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const double*     beta,
                                  double*           C,
                                  int               ldc);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const hipComplex* alpha,
                                  const hipComplex* A,
                                  int               lda,
                                  const hipComplex* cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const hipComplex* beta,
                                  hipComplex*       C,
                                  int               ldc);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgemmi(hipsparseHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  int                     k,
                                  int                     nnz,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A,
                                  int                     lda,
                                  const hipDoubleComplex* cscValB,
                                  const int*              cscColPtrB,
                                  const int*              cscRowIndB,
                                  const hipDoubleComplex* beta,
                                  hipDoubleComplex*       C,
                                  int                     ldc);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif