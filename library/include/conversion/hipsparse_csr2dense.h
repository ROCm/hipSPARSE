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
#ifndef HIPSPARSE_CONVERSION_HIPSPARSE_CSR2DENSE_H
#define HIPSPARSE_CONVERSION_HIPSPARSE_CSR2DENSE_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup conv_module
*  \brief
*  This function converts the sparse matrix in CSR format into a dense matrix.
*  It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      float*                    A,
                                      int                       ld);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      double*                   A,
                                      int                       ld);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      hipComplex*               A,
                                      int                       ld);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      hipDoubleComplex*         A,
                                      int                       ld);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CONVERSION_HIPSPARSE_CSR2DENSE_H */