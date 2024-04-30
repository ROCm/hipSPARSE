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
#ifndef HIPSPARSE_CONVERSION_HIPSPARSE_PRUNE_DENSE2CSR_BY_PERCENTAGE_H
#define HIPSPARSE_CONVERSION_HIPSPARSE_PRUNE_DENSE2CSR_BY_PERCENTAGE_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  This function computes the size of the user allocated temporary storage buffer used
*  when converting and pruning by percentage a dense matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
*  \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
*  by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
*  at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m*n*(percentage/100)) - 1
*    pos = min(pos, m*n-1)
*    pos = max(pos, 0)
*    threshold = sorted_A[pos]
*  \f]
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \p hipsparseXpruneDense2csr. It is executed asynchronously with respect to the host
*  and may return control to the application on the host before the entire result is
*  ready.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t handle,
                                                                  int               m,
                                                                  int               n,
                                                                  const float*      A,
                                                                  int               lda,
                                                                  float             percentage,
                                                                  const hipsparseMatDescr_t descr,
                                                                  const float*              csrVal,
                                                                  const int*  csrRowPtr,
                                                                  const int*  csrColInd,
                                                                  pruneInfo_t info,
                                                                  size_t*     bufferSize);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t handle,
                                                                  int               m,
                                                                  int               n,
                                                                  const double*     A,
                                                                  int               lda,
                                                                  double            percentage,
                                                                  const hipsparseMatDescr_t descr,
                                                                  const double*             csrVal,
                                                                  const int*  csrRowPtr,
                                                                  const int*  csrColInd,
                                                                  pruneInfo_t info,
                                                                  size_t*     bufferSize);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  This function computes the size of the user allocated temporary storage buffer used
*  when converting and pruning by percentage a dense matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \p hipsparseXpruneDense2csrByPercentage_bufferSizeExt which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
*  \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
*  by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
*  at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m*n*(percentage/100)) - 1
*    pos = min(pos, m*n-1)
*    pos = max(pos, 0)
*    threshold = sorted_A[pos]
*  \f]
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \p hipsparseXpruneDense2csr. It is executed asynchronously with respect to the host
*  and may return control to the application on the host before the entire result is
*  ready.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseSpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const float*              A,
                                                       int                       lda,
                                                       float                     percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       const float*              csrVal,
                                                       const int*                csrRowPtr,
                                                       const int*                csrColInd,
                                                       pruneInfo_t               info,
                                                       size_t*                   bufferSize);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseDpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const double*             A,
                                                       int                       lda,
                                                       double                    percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       const double*             csrVal,
                                                       const int*                csrRowPtr,
                                                       const int*                csrColInd,
                                                       pruneInfo_t               info,
                                                       size_t*                   bufferSize);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero elements per row and the total number of
*  nonzero elements in a dense matrix when converting and pruning by percentage a dense
*  matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
*  \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
*  by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
*  at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m*n*(percentage/100)) - 1
*    pos = min(pos, m*n-1)
*    pos = max(pos, 0)
*    threshold = sorted_A[pos]
*  \f]
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \p hipsparseXpruneDense2csr. The routine does support asynchronous execution if the
*  pointer mode is set to device.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const float*              A,
                                                          int                       lda,
                                                          float                     percentage,
                                                          const hipsparseMatDescr_t descr,
                                                          int*                      csrRowPtr,
                                                          int*        nnzTotalDevHostPtr,
                                                          pruneInfo_t info,
                                                          void*       buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const double*             A,
                                                          int                       lda,
                                                          double                    percentage,
                                                          const hipsparseMatDescr_t descr,
                                                          int*                      csrRowPtr,
                                                          int*        nnzTotalDevHostPtr,
                                                          pruneInfo_t info,
                                                          void*       buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero elements per row and the total number of
*  nonzero elements in a dense matrix when converting and pruning by percentage a dense
*  matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
*  \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
*  by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
*  at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m*n*(percentage/100)) - 1
*    pos = min(pos, m*n-1)
*    pos = max(pos, 0)
*    threshold = sorted_A[pos]
*  \f]
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \p hipsparseXpruneDense2csr. The routine does support asynchronous execution if the
*  pointer mode is set to device.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const float*              A,
                                                       int                       lda,
                                                       float                     percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       float*                    csrVal,
                                                       const int*                csrRowPtr,
                                                       int*                      csrColInd,
                                                       pruneInfo_t               info,
                                                       void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const double*             A,
                                                       int                       lda,
                                                       double                    percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       double*                   csrVal,
                                                       const int*                csrRowPtr,
                                                       int*                      csrColInd,
                                                       pruneInfo_t               info,
                                                       void*                     buffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CONVERSION_HIPSPARSE_PRUNE_DENSE2CSR_BY_PERCENTAGE_H */