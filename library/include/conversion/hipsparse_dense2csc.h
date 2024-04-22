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
#ifndef HIPSPARSE_CONVERSION_HIPSPARSE_DENSE2CSC_H
#define HIPSPARSE_CONVERSION_HIPSPARSE_DENSE2CSC_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup conv_module
*  \brief
*
*  This function converts the matrix A in dense format into a sparse matrix in CSC format.
*  All the parameters are assumed to have been pre-allocated by the user and the arrays are filled in based on nnz_per_columns, which can be pre-computed with hipsparseXnnz().
*  It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              A,
                                      int                       ld,
                                      const int*                nnz_per_columns,
                                      float*                    csc_val,
                                      int*                      csc_row_ind,
                                      int*                      csc_col_ptr);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             A,
                                      int                       ld,
                                      const int*                nnz_per_columns,
                                      double*                   csc_val,
                                      int*                      csc_row_ind,
                                      int*                      csc_col_ptr);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         A,
                                      int                       ld,
                                      const int*                nnz_per_columns,
                                      hipComplex*               csc_val,
                                      int*                      csc_row_ind,
                                      int*                      csc_col_ptr);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   A,
                                      int                       ld,
                                      const int*                nnz_per_columns,
                                      hipDoubleComplex*         csc_val,
                                      int*                      csc_row_ind,
                                      int*                      csc_col_ptr);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CONVERSION_HIPSPARSE_DENSE2CSC_H */