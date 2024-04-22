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
#ifndef HIPSPARSE_PRECOND_HIPSPARSE_BSRIC0_H
#define HIPSPARSE_PRECOND_HIPSPARSE_BSRIC0_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p hipsparseXbsric02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
 *  structural or numerical zero has been found during hipsparseXbsric02_analysis() or
 *  hipsparseXbsric02() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is
 *  stored in \p position, using same index base as the BSR matrix.
 *
 *  \p position can be in host or device memory. If no zero pivot has been found,
 *  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
 *
 *  \note
 *  If a zero pivot is found, \p position=j means that either the diagonal block \p A(j,j)
 *  is missing (structural zero) or the diagonal block \p A(j,j) is not positive definite
 *  (numerical zero).
 *
 *  \note \p hipsparseXbsric02_zeroPivot is a blocking function. It might influence
 *  performance negatively.
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXbsric02_zeroPivot(hipsparseHandle_t handle, bsric02Info_t info, int* position);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p hipsparseXbsric02_bufferSize returns the size of the temporary storage buffer
 *  in bytes that is required by hipsparseXbsric02_analysis() and hipsparseXbsric02(). 
 *  The temporary storage buffer must be allocated by the user.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               float*                    bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               double*                   bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               hipComplex*               bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               hipDoubleComplex*         bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p hipsparseXbsric02_analysis performs the analysis step for hipsparseXbsric02().
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
hipsparseStatus_t hipsparseSbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p hipsparseXbsric02 computes the incomplete Cholesky factorization with 0 fill-ins
 *  and no pivoting of a sparse \f$mb \times mb\f$ BSR matrix \f$A\f$, such that
 *  \f[
 *    A \approx LL^T
 *  \f]
 *
 *  \p hipsparseXbsric02 requires a user allocated temporary buffer. Its size is returned
 *  by hipsparseXbsric02_bufferSize(). Furthermore, analysis meta data is required. It
 *  can be obtained by hipsparseXbsric02_analysis(). \p hipsparseXbsric02 reports the
 *  first zero pivot (either numerical or structural zero). The zero pivot status can be
 *  obtained by calling hipsparseXbsric02_zeroPivot().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    float*                    bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    double*                   bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    hipComplex*               bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    hipDoubleComplex*         bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_PRECOND_HIPSPARSE_BSRIC0_H */