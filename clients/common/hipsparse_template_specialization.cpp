/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "hipsparse.hpp"

#include <hipsparse.h>

namespace hipsparse {

template <>
hipsparseStatus_t hipsparseXaxpyi(hipsparseHandle_t handle,
                                  int nnz,
                                  const float* alpha,
                                  const float* x_val,
                                  const int* x_ind,
                                  float* y,
                                  hipsparseIndexBase_t idx_base)
{
    return hipsparseSaxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

template <>
hipsparseStatus_t hipsparseXaxpyi(hipsparseHandle_t handle,
                                  int nnz,
                                  const double* alpha,
                                  const double* x_val,
                                  const int* x_ind,
                                  double* y,
                                  hipsparseIndexBase_t idx_base)
{
    return hipsparseDaxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

template <>
hipsparseStatus_t hipsparseXdoti(hipsparseHandle_t handle,
                                 int nnz,
                                 const float* x_val,
                                 const int* x_ind,
                                 const float* y,
                                 float* result,
                                 hipsparseIndexBase_t idx_base)
{
    return hipsparseSdoti(handle, nnz, x_val, x_ind, y, result, idx_base);
}

template <>
hipsparseStatus_t hipsparseXdoti(hipsparseHandle_t handle,
                                 int nnz,
                                 const double* x_val,
                                 const int* x_ind,
                                 const double* y,
                                 double* result,
                                 hipsparseIndexBase_t idx_base)
{
    return hipsparseDdoti(handle, nnz, x_val, x_ind, y, result, idx_base);
}

template <>
hipsparseStatus_t hipsparseXgthr(hipsparseHandle_t handle,
                                 int nnz,
                                 const float* y,
                                 float* x_val,
                                 const int* x_ind,
                                 hipsparseIndexBase_t idx_base)
{
    return hipsparseSgthr(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
hipsparseStatus_t hipsparseXgthr(hipsparseHandle_t handle,
                                 int nnz,
                                 const double* y,
                                 double* x_val,
                                 const int* x_ind,
                                 hipsparseIndexBase_t idx_base)
{
    return hipsparseDgthr(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
hipsparseStatus_t hipsparseXgthrz(hipsparseHandle_t handle,
                                  int nnz,
                                  float* y,
                                  float* x_val,
                                  const int* x_ind,
                                  hipsparseIndexBase_t idx_base)
{
    return hipsparseSgthrz(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
hipsparseStatus_t hipsparseXgthrz(hipsparseHandle_t handle,
                                  int nnz,
                                  double* y,
                                  double* x_val,
                                  const int* x_ind,
                                  hipsparseIndexBase_t idx_base)
{
    return hipsparseDgthrz(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
hipsparseStatus_t hipsparseXroti(hipsparseHandle_t handle,
                                 int nnz,
                                 float* x_val,
                                 const int* x_ind,
                                 float* y,
                                 const float* c,
                                 const float* s,
                                 hipsparseIndexBase_t idx_base)
{
    return hipsparseSroti(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}

template <>
hipsparseStatus_t hipsparseXroti(hipsparseHandle_t handle,
                                 int nnz,
                                 double* x_val,
                                 const int* x_ind,
                                 double* y,
                                 const double* c,
                                 const double* s,
                                 hipsparseIndexBase_t idx_base)
{
    return hipsparseDroti(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}

template <>
hipsparseStatus_t hipsparseXsctr(hipsparseHandle_t handle,
                                 int nnz,
                                 const float* x_val,
                                 const int* x_ind,
                                 float* y,
                                 hipsparseIndexBase_t idx_base)
{
    return hipsparseSsctr(handle, nnz, x_val, x_ind, y, idx_base);
}

template <>
hipsparseStatus_t hipsparseXsctr(hipsparseHandle_t handle,
                                 int nnz,
                                 const double* x_val,
                                 const int* x_ind,
                                 double* y,
                                 hipsparseIndexBase_t idx_base)
{
    return hipsparseDsctr(handle, nnz, x_val, x_ind, y, idx_base);
}

template <>
hipsparseStatus_t hipsparseXcoomv(hipsparseHandle_t handle,
                                  hipsparseOperation_t trans,
                                  int m,
                                  int n,
                                  int nnz,
                                  const float* alpha,
                                  const hipsparseMatDescr_t descr,
                                  const float* coo_val,
                                  const int* coo_row_ind,
                                  const int* coo_col_ind,
                                  const float* x,
                                  const float* beta,
                                  float* y)
{
    return hipsparseScoomv(
        handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
}

template <>
hipsparseStatus_t hipsparseXcoomv(hipsparseHandle_t handle,
                                  hipsparseOperation_t trans,
                                  int m,
                                  int n,
                                  int nnz,
                                  const double* alpha,
                                  const hipsparseMatDescr_t descr,
                                  const double* coo_val,
                                  const int* coo_row_ind,
                                  const int* coo_col_ind,
                                  const double* x,
                                  const double* beta,
                                  double* y)
{
    return hipsparseDcoomv(
        handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
}

template <>
hipsparseStatus_t hipsparseXcsrmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t trans,
                                  int m,
                                  int n,
                                  int nnz,
                                  const float* alpha,
                                  const hipsparseMatDescr_t descr,
                                  const float* csr_val,
                                  const int* csr_row_ptr,
                                  const int* csr_col_ind,
                                  const float* x,
                                  const float* beta,
                                  float* y)
{
    return hipsparseScsrmv(
        handle, trans, m, n, nnz, alpha, descr, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);
}

template <>
hipsparseStatus_t hipsparseXcsrmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t trans,
                                  int m,
                                  int n,
                                  int nnz,
                                  const double* alpha,
                                  const hipsparseMatDescr_t descr,
                                  const double* csr_val,
                                  const int* csr_row_ptr,
                                  const int* csr_col_ind,
                                  const double* x,
                                  const double* beta,
                                  double* y)
{
    return hipsparseDcsrmv(
        handle, trans, m, n, nnz, alpha, descr, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);
}

template <>
hipsparseStatus_t hipsparseXellmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t trans,
                                  int m,
                                  int n,
                                  const float* alpha,
                                  const hipsparseMatDescr_t descr,
                                  const float* ell_val,
                                  const int* ell_col_ind,
                                  int ell_width,
                                  const float* x,
                                  const float* beta,
                                  float* y)
{
    return hipsparseSellmv(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}

template <>
hipsparseStatus_t hipsparseXellmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t trans,
                                  int m,
                                  int n,
                                  const double* alpha,
                                  const hipsparseMatDescr_t descr,
                                  const double* ell_val,
                                  const int* ell_col_ind,
                                  int ell_width,
                                  const double* x,
                                  const double* beta,
                                  double* y)
{
    return hipsparseDellmv(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}

template <>
hipsparseStatus_t hipsparseXhybmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t trans,
                                  const float* alpha,
                                  const hipsparseMatDescr_t descr,
                                  const hipsparseHybMat_t hyb,
                                  const float* x,
                                  const float* beta,
                                  float* y)
{
    return hipsparseShybmv(handle, trans, alpha, descr, hyb, x, beta, y);
}

template <>
hipsparseStatus_t hipsparseXhybmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t trans,
                                  const double* alpha,
                                  const hipsparseMatDescr_t descr,
                                  const hipsparseHybMat_t hyb,
                                  const double* x,
                                  const double* beta,
                                  double* y)
{
    return hipsparseDhybmv(handle, trans, alpha, descr, hyb, x, beta, y);
}

template <>
hipsparseStatus_t hipsparseXcsrmm2(hipsparseHandle_t handle,
                                   hipsparseOperation_t trans_A,
                                   hipsparseOperation_t trans_B,
                                   int m,
                                   int n,
                                   int k,
                                   int nnz,
                                   const float* alpha,
                                   const hipsparseMatDescr_t descr,
                                   const float* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   const float* B,
                                   int ldb,
                                   const float* beta,
                                   float* C,
                                   int ldc)
{
    return hipsparseScsrmm2(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
hipsparseStatus_t hipsparseXcsrmm2(hipsparseHandle_t handle,
                                   hipsparseOperation_t trans_A,
                                   hipsparseOperation_t trans_B,
                                   int m,
                                   int n,
                                   int k,
                                   int nnz,
                                   const double* alpha,
                                   const hipsparseMatDescr_t descr,
                                   const double* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   const double* B,
                                   int ldb,
                                   const double* beta,
                                   double* C,
                                   int ldc)
{
    return hipsparseDcsrmm2(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
hipsparseStatus_t hipsparseXcsr2csc(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    int nnz,
                                    const float* csr_val,
                                    const int* csr_row_ptr,
                                    const int* csr_col_ind,
                                    float* csc_val,
                                    int* csc_row_ind,
                                    int* csc_col_ptr,
                                    hipsparseAction_t copy_values,
                                    hipsparseIndexBase_t idx_base)
{
    return hipsparseScsr2csc(handle,
                             m,
                             n,
                             nnz,
                             csr_val,
                             csr_row_ptr,
                             csr_col_ind,
                             csc_val,
                             csc_row_ind,
                             csc_col_ptr,
                             copy_values,
                             idx_base);
}

template <>
hipsparseStatus_t hipsparseXcsr2csc(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    int nnz,
                                    const double* csr_val,
                                    const int* csr_row_ptr,
                                    const int* csr_col_ind,
                                    double* csc_val,
                                    int* csc_row_ind,
                                    int* csc_col_ptr,
                                    hipsparseAction_t copy_values,
                                    hipsparseIndexBase_t idx_base)
{
    return hipsparseDcsr2csc(handle,
                             m,
                             n,
                             nnz,
                             csr_val,
                             csr_row_ptr,
                             csr_col_ind,
                             csc_val,
                             csc_row_ind,
                             csc_col_ptr,
                             copy_values,
                             idx_base);
}

template <>
hipsparseStatus_t hipsparseXcsr2ell(hipsparseHandle_t handle,
                                    int m,
                                    const hipsparseMatDescr_t csr_descr,
                                    const float* csr_val,
                                    const int* csr_row_ptr,
                                    const int* csr_col_ind,
                                    const hipsparseMatDescr_t ell_descr,
                                    int ell_width,
                                    float* ell_val,
                                    int* ell_col_ind)
{
    return hipsparseScsr2ell(handle,
                             m,
                             csr_descr,
                             csr_val,
                             csr_row_ptr,
                             csr_col_ind,
                             ell_descr,
                             ell_width,
                             ell_val,
                             ell_col_ind);
}

template <>
hipsparseStatus_t hipsparseXcsr2ell(hipsparseHandle_t handle,
                                    int m,
                                    const hipsparseMatDescr_t csr_descr,
                                    const double* csr_val,
                                    const int* csr_row_ptr,
                                    const int* csr_col_ind,
                                    const hipsparseMatDescr_t ell_descr,
                                    int ell_width,
                                    double* ell_val,
                                    int* ell_col_ind)
{
    return hipsparseDcsr2ell(handle,
                             m,
                             csr_descr,
                             csr_val,
                             csr_row_ptr,
                             csr_col_ind,
                             ell_descr,
                             ell_width,
                             ell_val,
                             ell_col_ind);
}

template <>
hipsparseStatus_t hipsparseXcsr2hyb(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    const hipsparseMatDescr_t descr,
                                    const float* csr_val,
                                    const int* csr_row_ptr,
                                    const int* csr_col_ind,
                                    hipsparseHybMat_t hyb,
                                    int user_ell_width,
                                    hipsparseHybPartition_t partition_type)
{
    return hipsparseScsr2hyb(handle,
                             m,
                             n,
                             descr,
                             csr_val,
                             csr_row_ptr,
                             csr_col_ind,
                             hyb,
                             user_ell_width,
                             partition_type);
}

template <>
hipsparseStatus_t hipsparseXcsr2hyb(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    const hipsparseMatDescr_t descr,
                                    const double* csr_val,
                                    const int* csr_row_ptr,
                                    const int* csr_col_ind,
                                    hipsparseHybMat_t hyb,
                                    int user_ell_width,
                                    hipsparseHybPartition_t partition_type)
{
    return hipsparseDcsr2hyb(handle,
                             m,
                             n,
                             descr,
                             csr_val,
                             csr_row_ptr,
                             csr_col_ind,
                             hyb,
                             user_ell_width,
                             partition_type);
}

} // namespace hipsparse
