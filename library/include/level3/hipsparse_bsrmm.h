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

/*! \ingroup level3_module
 *  \brief Sparse matrix dense matrix multiplication using BSR storage format
 *
 *  \details
 *  \p hipsparseXbsrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$mb \times kb\f$
 *  matrix \f$A\f$, defined in BSR storage format, and the dense \f$k \times n\f$
 *  matrix \f$B\f$ (where \f$k = block\_dim \times kb\f$) and adds the result to the dense
 *  \f$m \times n\f$ matrix \f$C\f$ (where \f$m = block\_dim \times mb\f$) that
 *  is multiplied by the scalar \f$\beta\f$, such that
 *  \f[
 *    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
 *    \end{array}
 *    \right.
 *  \f]
 *  and
 *  \f[
 *    op(B) = \left\{
 *    \begin{array}{ll}
 *        B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
 *        B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans_A == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
 *
 *  \par Example
 *  \code{.c}
 *      // hipSPARSE handle
 *      hipsparseHandle_t handle;
 *      hipsparseCreate(&handle);
 *
 *      //     1 2 0 3 0 0
 *      // A = 0 4 5 0 0 0
 *      //     0 0 0 7 8 0
 *      //     0 0 1 2 4 1
 *
 *      int block_dim = 2;
 *      int mb   = 2;
 *      int kb   = 3;
 *      int nnzb = 4;
 *      hipsparseDirection_t dir = HIPSPARSE_DIRECTION_ROW;
 *
 *      int hbsr_row_ptr[2 + 1]   = {0, 2, 4};
 *      int hbsr_col_ind[4]       = {0, 1, 1, 2};
 *      float hbsr_val[4 * 2 * 2] = {1, 2, 0, 4, 0, 3, 5, 0, 0, 7, 1, 2, 8, 0, 4, 1};
 *
 *      // Set dimension n of B
 *      int n = 3;
 *      int m = mb * block_dim;
 *      int k = kb * block_dim;
 *
 *      // Allocate and generate dense matrix B (k x n)
 *      float hB[6 * 3] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 
 *                      11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f};
 *
 *      int* dbsr_row_ptr = NULL;
 *      int* dbsr_col_ind = NULL;
 *      float* dbsr_val = NULL;
 *      hipMalloc((void**)&dbsr_row_ptr, sizeof(int) * (mb + 1));
 *      hipMalloc((void**)&dbsr_col_ind, sizeof(int) * nnzb);
 *      hipMalloc((void**)&dbsr_val, sizeof(float) * nnzb * block_dim * block_dim);
 *      hipMemcpy(dbsr_row_ptr, hbsr_row_ptr, sizeof(int) * (mb + 1), hipMemcpyHostToDevice);
 *      hipMemcpy(dbsr_col_ind, hbsr_col_ind, sizeof(int) * nnzb, hipMemcpyHostToDevice);
 *      hipMemcpy(dbsr_val, hbsr_val, sizeof(float) * nnzb * block_dim * block_dim, hipMemcpyHostToDevice);
 *
 *      // Copy B to the device
 *      float* dB;
 *      hipMalloc((void**)&dB, sizeof(float) * k * n);
 *      hipMemcpy(dB, hB, sizeof(float) * k * n, hipMemcpyHostToDevice);
 *
 *      // alpha and beta
 *      float alpha = 1.0f;
 *      float beta  = 0.0f;
 *
 *      // Allocate memory for the resulting matrix C
 *      float* dC;
 *      hipMalloc((void**)&dC, sizeof(float) * m * n);
 *
 *      // Matrix descriptor
 *      hipsparseMatDescr_t descr;
 *      hipsparseCreateMatDescr(&descr);
 *
 *      // Perform the matrix multiplication
 *      hipsparseSbsrmm(handle,
 *                      dir,
 *                      HIPSPARSE_OPERATION_NON_TRANSPOSE,
 *                      HIPSPARSE_OPERATION_NON_TRANSPOSE,
 *                      mb,
 *                      n,
 *                      kb,
 *                      nnzb,
 *                      &alpha,
 *                      descr,
 *                      dbsr_val,
 *                      dbsr_row_ptr,
 *                      dbsr_col_ind,
 *                      block_dim,
 *                      dB,
 *                      k,
 *                      &beta,
 *                      dC,
 *                      m);
 *
 *      // Copy results to host
 *      float hC[6 * 3];
 *      hipMemcpy(hC, dC, sizeof(float) * m * n, hipMemcpyDeviceToHost);
 *
 *      hipFree(dbsr_row_ptr);
 *      hipFree(dbsr_col_ind);
 *      hipFree(dbsr_val);
 *      hipFree(dB);
 *      hipFree(dC);
 *  \endcode
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const float*              B,
                                  int                       ldb,
                                  const float*              beta,
                                  float*                    C,
                                  int                       ldc);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const double*             B,
                                  int                       ldb,
                                  const double*             beta,
                                  double*                   C,
                                  int                       ldc);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const hipComplex*         B,
                                  int                       ldb,
                                  const hipComplex*         beta,
                                  hipComplex*               C,
                                  int                       ldc);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const hipDoubleComplex*   B,
                                  int                       ldb,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         C,
                                  int                       ldc);
/**@}*/

#ifdef __cplusplus
}
#endif