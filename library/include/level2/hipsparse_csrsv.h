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
/*! \ingroup level2_module
*  \brief Sparse triangular solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsv2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during hipsparseScsrsv2_solve(),
*  hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() or hipsparseZcsrsv2_solve()
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position,
*  using same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXcsrsv2_zeroPivot is a blocking function. It might influence
*  performance negatively.
*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXcsrsv2_zeroPivot(hipsparseHandle_t handle, csrsv2Info_t info, int* position);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level2_module
*  \brief Sparse triangular solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsv2_bufferSize returns the size of the temporary storage buffer in bytes 
*  that is required by hipsparseScsrsv2_analysis(), hipsparseDcsrsv2_analysis(),
*  hipsparseCcsrsv2_analysis(), hipsparseZcsrsv2_analysis(), hipsparseScsrsv2_solve(),
*  hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve(). The
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
/**@}*/
#endif

/*! \ingroup level2_module
*  \brief Sparse triangular solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsv2_bufferSizeExt returns the size of the temporary storage buffer in bytes 
*  that is required by hipsparseScsrsv2_analysis(), hipsparseDcsrsv2_analysis(),
*  hipsparseCcsrsv2_analysis(), hipsparseZcsrsv2_analysis(), hipsparseScsrsv2_solve(),
*  hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve(). The
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level2_module
*  \brief Sparse triangular solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsv2_analysis performs the analysis step for hipsparseScsrsv2_solve(),
*  hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level2_module
*  \brief Sparse triangular solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsv2_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution vector
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
*  \p hipsparseXcsrsv2_solve requires a user allocated temporary buffer. Its size is
*  returned by hipsparseXcsrsv2_bufferSize() or hipsparseXcsrsv2_bufferSizeExt().
*  Furthermore, analysis meta data is required. It can be obtained by
*  hipsparseXcsrsv2_analysis(). \p hipsparseXcsrsv2_solve reports the first zero pivot
*  (either numerical or structural zero). The zero pivot status can be checked calling
*  hipsparseXcsrsv2_zeroPivot(). If
*  \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
*  reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  hipsparseXcsrsort().
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
*      // alpha * ( 1.0  0.0  2.0  0.0 ) * ( x_0 ) = ( 32.0 )
*      //         ( 3.0  2.0  4.0  1.0 ) * ( x_1 ) = ( 14.7 )
*      //         ( 5.0  6.0  1.0  3.0 ) * ( x_2 ) = ( 33.6 )
*      //         ( 7.0  0.0  8.0  0.6 ) * ( x_3 ) = ( 10.0 )
*
*      int m = 4;
*      int nnz = 13;
*
*      // CSR row pointers
*      int hcsr_row_ptr[5] = {0, 2, 6, 10, 13};
*
*      // CSR column indices
*      int hcsr_col_ind[13] = {0, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 3};
*
*      // CSR values
*      double hcsr_val[13] = {1.0, 2.0, 3.0, 2.0, 4.0, 1.0, 5.0, 6.0, 1.0, 3.0, 7.0, 8.0, 0.6};
*
*      // Transposition of the matrix
*      hipsparseOperation_t trans = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*      hipsparseSolvePolicy_t policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
*
*      // Scalar alpha
*      double alpha = 1.0;
*
*      // f and x
*      double hf[4] = {32.0, 14.7, 33.6, 10.0};
*      double hx[4];
*
*      // Matrix descriptor
*      hipsparseMatDescr_t descr;
*      hipsparseCreateMatDescr(&descr);
*   
*      // Set index base on descriptor
*      hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO);
*
*      // Set fill mode on descriptor
*      hipsparseSetMatFillMode(descr, HIPSPARSE_FILL_MODE_LOWER);
*
*      // Set diag type on descriptor
*      hipsparseSetMatDiagType(descr, HIPSPARSE_DIAG_TYPE_UNIT);
*
*      // Csrsv info
*      csrsv2Info_t info;
*      hipsparseCreateCsrsv2Info(&info);
*
*      // Offload data to device
*      int* dcsr_row_ptr;
*      int* dcsr_col_ind;
*      double*        dcsr_val;
*      double*        df;
*      double*        dx;
*
*      hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1));
*      hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz);
*      hipMalloc((void**)&dcsr_val, sizeof(double) * nnz);
*      hipMalloc((void**)&df, sizeof(double) * m);
*      hipMalloc((void**)&dx, sizeof(double) * m);
*
*      hipMemcpy(dcsr_row_ptr, hcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*      hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(int) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dcsr_val, hcsr_val, sizeof(double) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(df, hf, sizeof(double) * m, hipMemcpyHostToDevice);
*
*      int bufferSize = 0;
*      hipsparseDcsrsv2_bufferSize(handle,
*                                  trans,
*                                  m,
*                                  nnz,
*                                  descr,
*                                  dcsr_val,
*                                  dcsr_row_ptr,
*                                  dcsr_col_ind,
*                                  info,
*                                  &bufferSize);
*
*      void* dbuffer = nullptr;
*      hipMalloc((void**)&dbuffer, bufferSize);
*
*      hipsparseDcsrsv2_analysis(handle,
*                                trans,
*                                m,
*                                nnz,
*                                descr,
*                                dcsr_val,
*                                dcsr_row_ptr,
*                                dcsr_col_ind,
*                                info,
*                                policy,
*                                dbuffer);
*
*      // Call dcsrsv to perform alpha * A * x = f
*      hipsparseDcsrsv2_solve(handle,
*                             trans,
*                             m,
*                             nnz,
*                             &alpha,
*                             descr,
*                             dcsr_val,
*                             dcsr_row_ptr,
*                             dcsr_col_ind,
*                             info,
*                             df,
*                             dx,
*                             policy,
*                             dbuffer);
*
*      // Copy result back to host
*      hipMemcpy(hx, dx, sizeof(double) * m, hipMemcpyDeviceToHost);
*
*      // Clear hipSPARSE
*      hipsparseDestroyMatDescr(descr);
*      hipsparseDestroyCsrsv2Info(info);
*      hipsparseDestroy(handle);
*
*      // Clear device memory
*      hipFree(dcsr_row_ptr);
*      hipFree(dcsr_col_ind);
*      hipFree(dcsr_val);
*      hipFree(df);
*      hipFree(dx);
*      hipFree(dbuffer);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const float*              f,
                                         float*                    x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const double*             f,
                                         double*                   x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const hipComplex*         f,
                                         hipComplex*               x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const hipDoubleComplex*   f,
                                         hipDoubleComplex*         x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif