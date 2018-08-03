/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _HIPSPARSE_HPP_
#define _HIPSPARSE_HPP_

#include <hipsparse.h>

namespace hipsparse {

template <typename T>
hipsparseStatus_t hipsparseXaxpyi(hipsparseHandle_t handle,
                                  int nnz,
                                  const T* alpha,
                                  const T* x_val,
                                  const int* x_ind,
                                  T* y,
                                  hipsparseIndexBase_t idx_base);

template <typename T>
hipsparseStatus_t hipsparseXdoti(hipsparseHandle_t handle,
                                 int nnz,
                                 const T* x_val,
                                 const int* x_ind,
                                 const T* y,
                                 T* result,
                                 hipsparseIndexBase_t idx_base);

template <typename T>
hipsparseStatus_t hipsparseXgthr(hipsparseHandle_t handle,
                                 int nnz,
                                 const T* y,
                                 T* x_val,
                                 const int* x_ind,
                                 hipsparseIndexBase_t idx_base);

template <typename T>
hipsparseStatus_t hipsparseXgthrz(hipsparseHandle_t handle,
                                  int nnz,
                                  T* y,
                                  T* x_val,
                                  const int* x_ind,
                                  hipsparseIndexBase_t idx_base);

template <typename T>
hipsparseStatus_t hipsparseXroti(hipsparseHandle_t handle,
                                 int nnz,
                                 T* x_val,
                                 const int* x_ind,
                                 T* y,
                                 const T* c,
                                 const T* s,
                                 hipsparseIndexBase_t idx_base);

template <typename T>
hipsparseStatus_t hipsparseXsctr(hipsparseHandle_t handle,
                                 int nnz,
                                 const T* x_val,
                                 const int* x_ind,
                                 T* y,
                                 hipsparseIndexBase_t idx_base);

template <typename T>
hipsparseStatus_t hipsparseXcoomv(hipsparseHandle_t handle,
                                  hipsparseOperation_t trans,
                                  int m,
                                  int n,
                                  int nnz,
                                  const T* alpha,
                                  const hipsparseMatDescr_t descr,
                                  const T* coo_val,
                                  const int* coo_row_ind,
                                  const int* coo_col_ind,
                                  const T* x,
                                  const T* beta,
                                  T* y);
template <typename T>
hipsparseStatus_t hipsparseXcsrmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t trans,
                                  int m,
                                  int n,
                                  int nnz,
                                  const T* alpha,
                                  const hipsparseMatDescr_t descr,
                                  const T* csr_val,
                                  const int* csr_row_ptr,
                                  const int* csr_col_ind,
                                  const T* x,
                                  const T* beta,
                                  T* y);

template <typename T>
hipsparseStatus_t hipsparseXellmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t trans,
                                  int m,
                                  int n,
                                  const T* alpha,
                                  const hipsparseMatDescr_t descr,
                                  const T* ell_val,
                                  const int* ell_col_ind,
                                  int ell_width,
                                  const T* x,
                                  const T* beta,
                                  T* y);

template <typename T>
hipsparseStatus_t hipsparseXhybmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t trans,
                                  const T* alpha,
                                  const hipsparseMatDescr_t descr,
                                  const hipsparseHybMat_t hyb,
                                  const T* x,
                                  const T* beta,
                                  T* y);

template <typename T>
hipsparseStatus_t hipsparseXcsrmm2(hipsparseHandle_t handle,
                                   hipsparseOperation_t transA,
                                   hipsparseOperation_t transB,
                                   int m,
                                   int n,
                                   int k,
                                   int nnz,
                                   const T* alpha,
                                   const hipsparseMatDescr_t descr,
                                   const T* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   const T* B,
                                   int ldb,
                                   const T* beta,
                                   T* C,
                                   int ldc);

template <typename T>
hipsparseStatus_t hipsparseXcsr2csc(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    int nnz,
                                    const T* csr_val,
                                    const int* csr_row_ptr,
                                    const int* csr_col_ind,
                                    T* csc_val,
                                    int* csc_row_ind,
                                    int* csc_col_ptr,
                                    hipsparseAction_t copy_values,
                                    hipsparseIndexBase_t idx_base);

template <typename T>
hipsparseStatus_t hipsparseXcsr2ell(hipsparseHandle_t handle,
                                    int m,
                                    const hipsparseMatDescr_t csr_descr,
                                    const T* csr_val,
                                    const int* csr_row_ptr,
                                    const int* csr_col_ind,
                                    const hipsparseMatDescr_t ell_descr,
                                    int ell_width,
                                    T* ell_val,
                                    int* ell_col_ind);

template <typename T>
hipsparseStatus_t hipsparseXcsr2hyb(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    const hipsparseMatDescr_t descr,
                                    const T* csr_val,
                                    const int* csr_row_ptr,
                                    const int* csr_col_ind,
                                    hipsparseHybMat_t hyb,
                                    int user_ell_width,
                                    hipsparseHybPartition_t partition_type);

template <typename T>
hipsparseStatus_t hipsparseXell2csr(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    const hipsparseMatDescr_t ell_descr,
                                    int ell_width,
                                    const T* ell_val,
                                    const int* ell_col_ind,
                                    const hipsparseMatDescr_t csr_descr,
                                    T* csr_val,
                                    const int* csr_row_ptr,
                                    int* csr_col_ind);

} // namespace hipsparse

#endif // _HIPSPARSE_HPP_
