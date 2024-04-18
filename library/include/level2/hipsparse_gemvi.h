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

/*! \ingroup level2_module
 *  \brief Dense matrix sparse vector multiplication
 *
 *  \details
 *  \p hipsparseXgemvi_bufferSize returns the size of the temporary storage buffer in bytes
 *  required by hipsparseXgemvi(). The temporary storage buffer must be allocated by the
 *  user.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes);
/**@}*/

/*! \ingroup level2_module
 *  \brief Dense matrix sparse vector multiplication
 *
 *  \details
 *  \p hipsparseXgemvi multiplies the scalar \f$\alpha\f$ with a dense \f$m \times n\f$
 *  matrix \f$A\f$ and the sparse vector \f$x\f$ and adds the result to the dense vector
 *  \f$y\f$ that is multiplied by the scalar \f$\beta\f$, such that
 *  \f[
 *    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
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
 *
 *  \p hipsparseXgemvi requires a user allocated temporary buffer. Its size is returned
 *  by hipsparseXgemvi_bufferSize().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const float*         alpha,
                                  const float*         A,
                                  int                  lda,
                                  int                  nnz,
                                  const float*         x,
                                  const int*           xInd,
                                  const float*         beta,
                                  float*               y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const double*        alpha,
                                  const double*        A,
                                  int                  lda,
                                  int                  nnz,
                                  const double*        x,
                                  const int*           xInd,
                                  const double*        beta,
                                  double*              y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const hipComplex*    alpha,
                                  const hipComplex*    A,
                                  int                  lda,
                                  int                  nnz,
                                  const hipComplex*    x,
                                  const int*           xInd,
                                  const hipComplex*    beta,
                                  hipComplex*          y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgemvi(hipsparseHandle_t       handle,
                                  hipsparseOperation_t    transA,
                                  int                     m,
                                  int                     n,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A,
                                  int                     lda,
                                  int                     nnz,
                                  const hipDoubleComplex* x,
                                  const int*              xInd,
                                  const hipDoubleComplex* beta,
                                  hipDoubleComplex*       y,
                                  hipsparseIndexBase_t    idxBase,
                                  void*                   pBuffer);
/**@}*/

#ifdef __cplusplus
}
#endif