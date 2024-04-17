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

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsv2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during hipsparseXbsrsv2_analysis() or
*  hipsparseXbsrsv2_solve() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
*  is stored in \p position, using same index base as the BSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXbsrsv2_zeroPivot is a blocking function. It might influence
*  performance negatively.
*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXbsrsv2_zeroPivot(hipsparseHandle_t handle, bsrsv2Info_t info, int* position);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsv2_bufferSize returns the size of the temporary storage buffer in bytes 
*  that is required by hipsparseXbsrsv2_analysis() and hipsparseXbsrsv2_solve(). The
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
/**@}*/
#endif

/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsv2_bufferSizeExt returns the size of the temporary storage buffer in bytes 
*  that is required by hipsparseXbsrsv2_analysis() and hipsparseXbsrsv2_solve(). The
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
/**@}*/


#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsv2_analysis performs the analysis step for hipsparseXbsrsv2_solve().
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsv2_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in BSR storage format, a dense solution vector
*  \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot y = \alpha \cdot x,
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
*  \p hipsparseXbsrsv2_solve requires a user allocated temporary buffer. Its size is
*  returned by hipsparseXbsrsv2_bufferSize() or hipsparseXbsrsv2_bufferSizeExt().
*  Furthermore, analysis meta data is required. It can be obtained by
*  hipsparseXbsrsv2_analysis(). \p hipsparseXbsrsv2_solve reports the first zero pivot
*  (either numerical or structural zero). The zero pivot status can be checked calling
*  hipsparseXbsrsv2_zeroPivot(). If
*  \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
*  reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
*
*  \note
*  The sparse BSR matrix has to be sorted.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE and
*  \p trans == \ref HIPSPARSE_OPERATION_TRANSPOSE is supported.
*
*  \par Example
*  \code{.c}
*      // hipSPARSE handle
*      hipsparseHandle_t handle;
*      hipsparseCreate(&handle);
*
*      // A = ( 1.0  0.0  0.0  0.0 )
*      //     ( 2.0  3.0  0.0  0.0 )
*      //     ( 4.0  5.0  6.0  0.0 )
*      //     ( 7.0  0.0  8.0  9.0 )
*      //
*      // with bsr_dim = 2
*      //
*      //      -------------------
*      //   = | 1.0 0.0 | 0.0 0.0 |
*      //     | 2.0 3.0 | 0.0 0.0 |
*      //      -------------------
*      //     | 4.0 5.0 | 6.0 0.0 |
*      //     | 7.0 0.0 | 8.0 9.0 |
*      //      -------------------
*
*      // Number of rows and columns
*      int m = 4;
*
*      // Number of block rows and block columns
*      int mb = 2;
*      int nb = 2;
*
*      // BSR block dimension
*      int bsr_dim = 2;
*
*      // Number of non-zero blocks
*      int nnzb = 3;
*
*      // BSR row pointers
*      int hbsr_row_ptr[3] = {0, 1, 3};
*
*      // BSR column indices
*      int hbsr_col_ind[3] = {0, 0, 1};
*
*      // BSR values
*      double hbsr_val[12] = {1.0, 2.0, 0.0, 3.0, 4.0, 7.0, 5.0, 0.0, 6.0, 8.0, 0.0, 9.0};
*
*      // Storage scheme of the BSR blocks
*      hipsparseDirection_t dir = HIPSPARSE_DIRECTION_COLUMN;
*
*      // Transposition of the matrix and rhs matrix
*      hipsparseOperation_t trans = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*
*      // Solve policy
*      hipsparseSolvePolicy_t solve_policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
*
*      // Scalar alpha and beta
*      double alpha = 3.7;
*
*      double hx[4] = {1, 2, 3, 4};
*      double hy[4];
*
*      // Offload data to device
*      int* dbsr_row_ptr;
*      int* dbsr_col_ind;
*      double* dbsr_val;
*      double* dx;
*      double* dy;
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
*
*      // Matrix descriptor
*      hipsparseMatDescr_t descr;
*      hipsparseCreateMatDescr(&descr);
*
*      // Matrix fill mode
*      hipsparseSetMatFillMode(descr, HIPSPARSE_FILL_MODE_LOWER);
*
*      // Matrix diagonal type
*      hipsparseSetMatDiagType(descr, HIPSPARSE_DIAG_TYPE_UNIT);
*
*      // Matrix info structure
*      bsrsv2Info_t info;
*      hipsparseCreateBsrsv2Info(&info);
*
*      // Obtain required buffer size
*      int buffer_size;
*      hipsparseDbsrsv2_bufferSize(handle,
*                                  dir,
*                                  trans,
*                                  mb,
*                                  nnzb,
*                                  descr,
*                                  dbsr_val,
*                                  dbsr_row_ptr,
*                                  dbsr_col_ind,
*                                  bsr_dim,
*                                  info,
*                                  &buffer_size);
*
*      // Allocate temporary buffer
*      void* dbuffer;
*      hipMalloc(&dbuffer, buffer_size);
*
*      // Perform analysis step
*      hipsparseDbsrsv2_analysis(handle,
*                                dir,
*                                trans,
*                                mb,
*                                nnzb,
*                                descr,
*                                dbsr_val,
*                                dbsr_row_ptr,
*                                dbsr_col_ind,
*                                bsr_dim,
*                                info,
*                                solve_policy,
*                                dbuffer);
*
*      // Call dbsrsm to perform lower triangular solve LX = B
*      hipsparseDbsrsv2_solve(handle,
*                             dir,
*                             trans,
*                             mb,
*                             nnzb,
*                             &alpha,
*                             descr,
*                             dbsr_val,
*                             dbsr_row_ptr,
*                             dbsr_col_ind,
*                             bsr_dim,
*                             info,
*                             dx,
*                             dy,
*                             solve_policy,
*                             dbuffer);
*
*      // Check for zero pivots
*      int    pivot;
*      hipsparseStatus_t status = hipsparseXbsrsv2_zeroPivot(handle, info, &pivot);
*
*      if(status == HIPSPARSE_STATUS_ZERO_PIVOT)
*      {
*          std::cout << "Found zero pivot in matrix row " << pivot << std::endl;
*      }
*
*      // Copy results back to the host
*      hipMemcpy(hy, dy, sizeof(double) * mb * bsr_dim, hipMemcpyDeviceToHost);
*
*      // Clear hipSPARSE
*      hipsparseDestroyBsrsv2Info(info);
*      hipsparseDestroyMatDescr(descr);
*      hipsparseDestroy(handle);
*
*      // Clear device memory
*      hipFree(dbsr_row_ptr);
*      hipFree(dbsr_col_ind);
*      hipFree(dbsr_val);
*      hipFree(dx);
*      hipFree(dy);
*      hipFree(dbuffer);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const float*              f,
                                         float*                    x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const double*             f,
                                         double*                   x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const hipComplex*         f,
                                         hipComplex*               x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const hipDoubleComplex*   f,
                                         hipDoubleComplex*         x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif