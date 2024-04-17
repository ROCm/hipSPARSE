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
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemmNnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting multiplied matrix C. It is assumed that \p csr_row_ptr_C has been allocated
*  with size \p m + 1.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Currently, only \p trans_A == \p trans_B == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is
*  supported.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgemmNnz(hipsparseHandle_t         handle,
                                       hipsparseOperation_t      transA,
                                       hipsparseOperation_t      transB,
                                       int                       m,
                                       int                       n,
                                       int                       k,
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
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm multiplies the sparse \f$m \times k\f$ matrix \f$A\f$, defined in
*  CSR storage format with the sparse \f$k \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format, and stores the result in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  defined in CSR storage format, such that
*  \f[
*    C := op(A) \cdot op(B),
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
*  It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
*  \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
*  \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
*  the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgemmNnz().
*
*  \note Currently, only \p trans_A == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*  \note Currently, only \p trans_B == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*  \note Please note, that for matrix products with more than 4096 non-zero entries per
*  row, additional temporary storage buffer is allocated by the algorithm.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const float*              csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const double*             csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipComplex*         csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipDoubleComplex*   csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
/**@}*/
#endif




#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm2_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes that is required by hipsparseXcsrgemm2Nnz() and hipsparseXcsrgemm2(). The 
*  temporary storage buffer must be allocated by the user.
*
*  \note
*  Please note, that for matrix products with more than 4096 non-zero entries per row,
*  additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const float*              alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const float*              beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const double*             alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const double*             beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const hipComplex*         alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const hipComplex*         beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const hipDoubleComplex*   alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const hipDoubleComplex*   beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm2Nnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting multiplied matrix C. It is assumed that \p csr_row_ptr_C has been allocated
*  with size \p m + 1.
*  The required buffer size can be obtained by hipsparseXcsrgemm2_bufferSizeExt().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgemm2Nnz(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        int                       k,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const int*                csrRowPtrB,
                                        const int*                csrColIndB,
                                        const hipsparseMatDescr_t descrD,
                                        int                       nnzD,
                                        const int*                csrRowPtrD,
                                        const int*                csrColIndD,
                                        const hipsparseMatDescr_t descrC,
                                        int*                      csrRowPtrC,
                                        int*                      nnzTotalDevHostPtr,
                                        const csrgemm2Info_t      info,
                                        void*                     pBuffer);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm2 multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times k\f$ matrix \f$A\f$, defined in CSR storage format, and the sparse
*  \f$k \times n\f$ matrix \f$B\f$, defined in CSR storage format, and adds the result
*  to the sparse \f$m \times n\f$ matrix \f$D\f$ that is multiplied by \f$\beta\f$. The
*  final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$, defined in CSR
*  storage format, such
*  that
*  \f[
*    C := \alpha \cdot A \cdot B + \beta \cdot D
*  \f]
*
*  It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
*  \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
*  \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
*  the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgemm2Nnz(). The
*  required buffer size for the computation can be obtained by
*  hipsparseXcsrgemm2_bufferSizeExt().
*
*  \note If \f$\alpha == 0\f$, then \f$C = \beta \cdot D\f$ will be computed.
*  \note If \f$\beta == 0\f$, then \f$C = \alpha \cdot A \cdot B\f$ will be computed.
*  \note \f$\alpha == beta == 0\f$ is invalid.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*  \note Please note, that for matrix products with more than 4096 non-zero entries per
*  row, additional temporary storage buffer is allocated by the algorithm.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const float*              alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const float*              csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const float*              csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const float*              beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const float*              csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     float*                    csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const double*             alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const double*             csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const double*             csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const double*             beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const double*             csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     double*                   csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const hipComplex*         alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipComplex*         csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipComplex*         csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const hipComplex*         beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const hipComplex*         csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     hipComplex*               csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const hipDoubleComplex*   alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipDoubleComplex*   csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipDoubleComplex*   csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const hipDoubleComplex*   beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const hipDoubleComplex*   csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     hipDoubleComplex*         csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif