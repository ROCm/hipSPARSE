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
*  \brief Sparse matrix vector multiplication using BSR storage format
*
*  \details
*  \p hipsparseXbsrmv multiplies the scalar \f$\alpha\f$ with a sparse
*  \f$(mb \cdot \text{block_dim}) \times (nb \cdot \text{block_dim})\f$
*  matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
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
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*
*  \par Example
*  \code{.c}
*      // hipSPARSE handle
*      hipsparseHandle_t handle;
*      hipsparseCreate(&handle);
*
*      // alpha * ( 1.0  0.0  2.0 ) * ( 1.0 ) + beta * ( 4.0 ) = (  31.1 )
*      //         ( 3.0  0.0  4.0 ) * ( 2.0 )          ( 5.0 ) = (  62.0 )
*      //         ( 5.0  6.0  0.0 ) * ( 3.0 )          ( 6.0 ) = (  70.7 )
*      //         ( 7.0  0.0  8.0 ) *                  ( 7.0 ) = ( 123.8 )
*
*      // BSR block dimension
*      int bsr_dim = 2;
*
*      // Number of block rows and columns
*      int mb = 2;
*      int nb = 2;
*
*      // Number of non-zero blocks
*      int nnzb = 4;
*
*      // BSR row pointers
*      int hbsr_row_ptr[3] = {0, 2, 4};
*
*      // BSR column indices
*      int hbsr_col_ind[4] = {0, 1, 0, 1};
*
*      // BSR values
*      double hbsr_val[16]
*        = {1.0, 3.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 5.0, 7.0, 6.0, 0.0, 0.0, 8.0, 0.0, 0.0};
*
*      // Block storage in column major
*      hipsparseDirection_t dir = HIPSPARSE_DIRECTION_COLUMN;
*
*      // Transposition of the matrix
*      hipsparseOperation_t trans = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*
*      // Scalar alpha and beta
*      double alpha = 3.7;
*      double beta  = 1.3;
*
*      // x and y
*      double hx[4] = {1.0, 2.0, 3.0, 0.0};
*      double hy[4] = {4.0, 5.0, 6.0, 7.0};
*
*      // Matrix descriptor
*      hipsparseMatDescr_t descr;
*      hipsparseCreateMatDescr(&descr);
*
*      // Offload data to device
*      int* dbsr_row_ptr;
*      int* dbsr_col_ind;
*      double*        dbsr_val;
*      double*        dx;
*      double*        dy;
*
*      hipMalloc((void**)&dbsr_row_ptr, sizeof(int) * (mb + 1));
*      hipMalloc((void**)&dbsr_col_ind, sizeof(int) * nnzb);
*      hipMalloc((void**)&dbsr_val, sizeof(double) * nnzb * bsr_dim * bsr_dim);
*      hipMalloc((void**)&dx, sizeof(double) * nb * bsr_dim);
*      hipMalloc((void**)&dy, sizeof(double) * mb * bsr_dim);
*
*      hipMemcpy(dbsr_row_ptr, hbsr_row_ptr, sizeof(int) * (mb + 1), hipMemcpyHostToDevice);
*      hipMemcpy(dbsr_col_ind, hbsr_col_ind, sizeof(int) * nnzb, hipMemcpyHostToDevice);
*      hipMemcpy(dbsr_val, hbsr_val, sizeof(double) * nnzb * bsr_dim * bsr_dim, hipMemcpyHostToDevice);
*      hipMemcpy(dx, hx, sizeof(double) * nb * bsr_dim, hipMemcpyHostToDevice);
*      hipMemcpy(dy, hy, sizeof(double) * mb * bsr_dim, hipMemcpyHostToDevice);
*
*      // Call dbsrmv to perform y = alpha * A x + beta * y
*      hipsparseDbsrmv(handle,
*                      dir,
*                      trans,
*                      mb,
*                      nb,
*                      nnzb,
*                      &alpha,
*                      descr,
*                      dbsr_val,
*                      dbsr_row_ptr,
*                      dbsr_col_ind,
*                      bsr_dim,
*                      dx,
*                      &beta,
*                      dy);
*
*      // Copy result back to host
*      hipMemcpy(hy, dy, sizeof(double) * mb * bsr_dim, hipMemcpyDeviceToHost);
*
*      // Clear hipSPARSE
*      hipsparseDestroyMatDescr(descr);
*      hipsparseDestroy(handle);
*
*      // Clear device memory
*      hipFree(dbsr_row_ptr);
*      hipFree(dbsr_col_ind);
*      hipFree(dbsr_val);
*      hipFree(dx);
*      hipFree(dy);
*  \endcode
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y);
/**@}*/

#ifdef __cplusplus
}
#endif
