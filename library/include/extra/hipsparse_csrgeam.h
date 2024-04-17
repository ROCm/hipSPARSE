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

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeamNnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting matrix C. It is assumed that \p csr_row_ptr_C has been allocated with
*  size \p m + 1.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgeamNnz(hipsparseHandle_t         handle,
                                       int                       m,
                                       int                       n,
                                       const hipsparseMatDescr_t descrA,
                                       int                       nnzA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       const hipsparseMatDescr_t descrB,
                                       int                       nnzB,
                                       const int*                csrRowPtrB,
                                       const int*                csrColIndB,
                                       const hipsparseMatDescr_t descrC,
                                       int*                      csrRowPtrC,
                                       int*                      nnzTotalDevHostPtr);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
*  scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format, and adds both resulting matrices to obtain the sparse
*  \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
*  \f[
*    C := \alpha \cdot A + \beta \cdot B.
*  \f]
*
*  It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
*  \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
*  \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
*  the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgeamNnz().
*
*  \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const float*              alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const float*              beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const float*              csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const double*             alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const double*             beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const double*             csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipComplex*         alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipComplex*         beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipComplex*         csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipDoubleComplex*   alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipDoubleComplex*   beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipDoubleComplex*   csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
/**@}*/
#endif

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam2_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes that is required by hipsparseXcsrgeam2Nnz() and hipsparseXcsrgeam2(). The 
*  temporary storage buffer must be allocated by the user.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const float*              alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const float*              csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const float*              beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const float*              csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const float*              csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const double*             alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const double*             csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const double*             beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const double*             csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const double*             csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const hipComplex*         alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const hipComplex*         csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const hipComplex*         beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const hipComplex*         csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const hipComplex*         csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const hipDoubleComplex*   alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const hipDoubleComplex*   csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const hipDoubleComplex*   beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const hipDoubleComplex*   csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const hipDoubleComplex*   csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);
/**@}*/

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam2Nnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting matrix C. It is assumed that \p csr_row_ptr_C has been allocated with
*  size \p m + 1.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgeam2Nnz(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const int*                csrSortedRowPtrA,
                                        const int*                csrSortedColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const int*                csrSortedRowPtrB,
                                        const int*                csrSortedColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        int*                      csrSortedRowPtrC,
                                        int*                      nnzTotalDevHostPtr,
                                        void*                     workspace);


/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam2 multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
*  scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format, and adds both resulting matrices to obtain the sparse
*  \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
*  \f[
*    C := \alpha \cdot A + \beta \cdot B.
*  \f]
*
*  It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
*  \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
*  \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
*  the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgeam2Nnz().
*
*  \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const float*              alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const float*              csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const float*              beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const float*              csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     float*                    csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const double*             alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const double*             csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const double*             beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const double*             csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     double*                   csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const hipComplex*         alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipComplex*         csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const hipComplex*         beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipComplex*         csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     hipComplex*               csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const hipDoubleComplex*   alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipDoubleComplex*   csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const hipDoubleComplex*   beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipDoubleComplex*   csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     hipDoubleComplex*         csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);
/**@}*/

#ifdef __cplusplus
}
#endif