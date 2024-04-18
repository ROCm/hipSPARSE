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

/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csr_bufferSize returns the size of the temporary buffer that
 *  is required by \p hipsparseXpruneCsr2csrNnz and hipsparseXpruneCsr2csr. The
 *  temporary storage buffer must be allocated by the user.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       n,
                                                    int                       nnzA,
                                                    const hipsparseMatDescr_t descrA,
                                                    const float*              csrValA,
                                                    const int*                csrRowPtrA,
                                                    const int*                csrColIndA,
                                                    const float*              threshold,
                                                    const hipsparseMatDescr_t descrC,
                                                    const float*              csrValC,
                                                    const int*                csrRowPtrC,
                                                    const int*                csrColIndC,
                                                    size_t*                   bufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       n,
                                                    int                       nnzA,
                                                    const hipsparseMatDescr_t descrA,
                                                    const double*             csrValA,
                                                    const int*                csrRowPtrA,
                                                    const int*                csrColIndA,
                                                    const double*             threshold,
                                                    const hipsparseMatDescr_t descrC,
                                                    const double*             csrValC,
                                                    const int*                csrRowPtrC,
                                                    const int*                csrColIndC,
                                                    size_t*                   bufferSize);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csr_bufferSizeExt returns the size of the temporary buffer that
 *  is required by \p hipsparseXpruneCsr2csrNnz and hipsparseXpruneCsr2csr. The
 *  temporary storage buffer must be allocated by the user.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       nnzA,
                                                       const hipsparseMatDescr_t descrA,
                                                       const float*              csrValA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const float*              threshold,
                                                       const hipsparseMatDescr_t descrC,
                                                       const float*              csrValC,
                                                       const int*                csrRowPtrC,
                                                       const int*                csrColIndC,
                                                       size_t*                   bufferSize);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       nnzA,
                                                       const hipsparseMatDescr_t descrA,
                                                       const double*             csrValA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const double*             threshold,
                                                       const hipsparseMatDescr_t descrC,
                                                       const double*             csrValC,
                                                       const int*                csrRowPtrC,
                                                       const int*                csrColIndC,
                                                       size_t*                   bufferSize);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csrNnz computes the number of nonzero elements per row and the total
 *  number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
 *  pruned from the matrix.
 *
 *  \note The routine does support asynchronous execution if the pointer mode is set to device.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrNnz(hipsparseHandle_t         handle,
                                            int                       m,
                                            int                       n,
                                            int                       nnzA,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrValA,
                                            const int*                csrRowPtrA,
                                            const int*                csrColIndA,
                                            const float*              threshold,
                                            const hipsparseMatDescr_t descrC,
                                            int*                      csrRowPtrC,
                                            int*                      nnzTotalDevHostPtr,
                                            void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrNnz(hipsparseHandle_t         handle,
                                            int                       m,
                                            int                       n,
                                            int                       nnzA,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrValA,
                                            const int*                csrRowPtrA,
                                            const int*                csrColIndA,
                                            const double*             threshold,
                                            const hipsparseMatDescr_t descrC,
                                            int*                      csrRowPtrC,
                                            int*                      nnzTotalDevHostPtr,
                                            void*                     buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
 *  that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
 *  The user first calls hipsparseXpruneCsr2csr_bufferSize() to determine the size of the buffer used
 *  by hipsparseXpruneCsr2csrNnz() and hipsparseXpruneCsr2csr() which the user then allocates. The user then
 *  allocates \p csr_row_ptr_C to have \p m+1 elements and then calls hipsparseXpruneCsr2csrNnz() which fills
 *  in the \p csr_row_ptr_C array stores then number of elements that are larger than the pruning threshold
 *  in \p nnz_total_dev_host_ptr. The user then calls hipsparseXpruneCsr2csr() to complete the conversion. It
 *  is executed asynchronously with respect to the host and may return control to the application on the host
 *  before the entire result is ready.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const float*              threshold,
                                         const hipsparseMatDescr_t descrC,
                                         float*                    csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const double*             threshold,
                                         const hipsparseMatDescr_t descrC,
                                         double*                   csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         void*                     buffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif