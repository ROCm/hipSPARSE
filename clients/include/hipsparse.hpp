/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#ifndef _HIPSPARSE_HPP_
#define _HIPSPARSE_HPP_

#include <hipsparse.h>

namespace hipsparse
{

    template <typename T>
    hipsparseStatus_t hipsparseXaxpyi(hipsparseHandle_t    handle,
                                      int                  nnz,
                                      const T*             alpha,
                                      const T*             x_val,
                                      const int*           x_ind,
                                      T*                   y,
                                      hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXdoti(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const T*             x_val,
                                     const int*           x_ind,
                                     const T*             y,
                                     T*                   result,
                                     hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXgthr(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const T*             y,
                                     T*                   x_val,
                                     const int*           x_ind,
                                     hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXgthrz(hipsparseHandle_t    handle,
                                      int                  nnz,
                                      T*                   y,
                                      T*                   x_val,
                                      const int*           x_ind,
                                      hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXroti(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     T*                   x_val,
                                     const int*           x_ind,
                                     T*                   y,
                                     const T*             c,
                                     const T*             s,
                                     hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXsctr(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const T*             x_val,
                                     const int*           x_ind,
                                     T*                   y,
                                     hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrmv(hipsparseHandle_t         handle,
                                      hipsparseOperation_t      trans,
                                      int                       m,
                                      int                       n,
                                      int                       nnz,
                                      const T*                  alpha,
                                      const hipsparseMatDescr_t descr,
                                      const T*                  csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      const T*                  x,
                                      const T*                  beta,
                                      T*                        y);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseOperation_t      transA,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  T*                        csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csrsv2Info_t              info,
                                                  int*                      pBufferSizeInBytes);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     hipsparseOperation_t      transA,
                                                     int                       m,
                                                     int                       nnz,
                                                     const hipsparseMatDescr_t descrA,
                                                     T*                        csrSortedValA,
                                                     const int*                csrSortedRowPtrA,
                                                     const int*                csrSortedColIndA,
                                                     csrsv2Info_t              info,
                                                     size_t*                   pBufferSize);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrsv2_analysis(hipsparseHandle_t         handle,
                                                hipsparseOperation_t      transA,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                const T*                  csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrsv2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrsv2_solve(hipsparseHandle_t         handle,
                                             hipsparseOperation_t      transA,
                                             int                       m,
                                             int                       nnz,
                                             const T*                  alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const T*                  csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csrsv2Info_t              info,
                                             const T*                  f,
                                             T*                        x,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXhybmv(hipsparseHandle_t         handle,
                                      hipsparseOperation_t      trans,
                                      const T*                  alpha,
                                      const hipsparseMatDescr_t descr,
                                      const hipsparseHybMat_t   hyb,
                                      const T*                  x,
                                      const T*                  beta,
                                      T*                        y);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrmm2(hipsparseHandle_t         handle,
                                       hipsparseOperation_t      transA,
                                       hipsparseOperation_t      transB,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       int                       nnz,
                                       const T*                  alpha,
                                       const hipsparseMatDescr_t descr,
                                       const T*                  csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       const T*                  B,
                                       int                       ldb,
                                       const T*                  beta,
                                       T*                        C,
                                       int                       ldc);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrgemm(hipsparseHandle_t         handle,
                                        hipsparseOperation_t      transA,
                                        hipsparseOperation_t      transB,
                                        int                       m,
                                        int                       n,
                                        int                       k,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const T*                  csrValA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const T*                  csrValB,
                                        const int*                csrRowPtrB,
                                        const int*                csrColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        T*                        csrValC,
                                        const int*                csrRowPtrC,
                                        int*                      csrColIndC);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       k,
                                                       const T*                  alpha,
                                                       const hipsparseMatDescr_t descrA,
                                                       int                       nnzA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const hipsparseMatDescr_t descrB,
                                                       int                       nnzB,
                                                       const int*                csrRowPtrB,
                                                       const int*                csrColIndB,
                                                       const T*                  beta,
                                                       const hipsparseMatDescr_t descrD,
                                                       int                       nnzD,
                                                       const int*                csrRowPtrD,
                                                       const int*                csrColIndD,
                                                       csrgemm2Info_t            info,
                                                       size_t* pBufferSizeInBytes);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrgemm2(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       k,
                                         const T*                  alpha,
                                         const hipsparseMatDescr_t descrA,
                                         int                       nnzA,
                                         const T*                  csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const hipsparseMatDescr_t descrB,
                                         int                       nnzB,
                                         const T*                  csrValB,
                                         const int*                csrRowPtrB,
                                         const int*                csrColIndB,
                                         const T*                  beta,
                                         const hipsparseMatDescr_t descrD,
                                         int                       nnzD,
                                         const T*                  csrValD,
                                         const int*                csrRowPtrD,
                                         const int*                csrColIndD,
                                         const hipsparseMatDescr_t descrC,
                                         T*                        csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         const csrgemm2Info_t      info,
                                         void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       nnz,
                                                    const hipsparseMatDescr_t descrA,
                                                    T*                        csrSortedValA,
                                                    const int*                csrSortedRowPtrA,
                                                    const int*                csrSortedColIndA,
                                                    csrilu02Info_t            info,
                                                    int*                      pBufferSizeInBytes);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       nnz,
                                                       const hipsparseMatDescr_t descrA,
                                                       T*                        csrSortedValA,
                                                       const int*                csrSortedRowPtrA,
                                                       const int*                csrSortedColIndA,
                                                       csrilu02Info_t            info,
                                                       size_t*                   pBufferSize);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrilu02_analysis(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  const T*                  csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csrilu02Info_t            info,
                                                  hipsparseSolvePolicy_t    policy,
                                                  void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrilu02(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         T*                        csrSortedValA_valM,
                                         /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                         const int*             csrSortedRowPtrA,
                                         const int*             csrSortedColIndA,
                                         csrilu02Info_t         info,
                                         hipsparseSolvePolicy_t policy,
                                         void*                  pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXcsr2csc(hipsparseHandle_t    handle,
                                        int                  m,
                                        int                  n,
                                        int                  nnz,
                                        const T*             csr_val,
                                        const int*           csr_row_ptr,
                                        const int*           csr_col_ind,
                                        T*                   csc_val,
                                        int*                 csc_row_ind,
                                        int*                 csc_col_ptr,
                                        hipsparseAction_t    copy_values,
                                        hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXcsr2hyb(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descr,
                                        const T*                  csr_val,
                                        const int*                csr_row_ptr,
                                        const int*                csr_col_ind,
                                        hipsparseHybMat_t         hyb,
                                        int                       user_ell_width,
                                        hipsparseHybPartition_t   partition_type);

} // namespace hipsparse

#endif // _HIPSPARSE_HPP_
