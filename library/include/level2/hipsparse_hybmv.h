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
#ifndef HIPSPARSE_LEVEL2_HIPSPARSE_HYBMV_H
#define HIPSPARSE_LEVEL2_HIPSPARSE_HYBMV_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using HYB storage format
*
*  \details
*  \p hipsparseXhybmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
*  matrix, defined in HYB storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*
*  \par Example
*  \code{.c}
*      // hipSPARSE handle
*      hipsparseHandle_t handle;
*      hipsparseCreate(&handle);
*
*      // A sparse matrix
*      // 1 0 3 4
*      // 0 0 5 1
*      // 0 2 0 0
*      // 4 0 0 8
*      int hAptr[5] = {0, 3, 5, 6, 8};
*      int hAcol[8] = {0, 2, 3, 2, 3, 1, 0, 3};
*      double hAval[8] = {1.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 8.0};
*
*      int m = 4;
*      int n = 4;
*      int nnz = 8;
*
*      double halpha = 1.0;
*      double hbeta  = 0.0;
*
*      double  hx[4] = {1.0, 2.0, 3.0, 4.0};
*      double  hy[4] = {4.0, 5.0, 6.0, 7.0};
*
*      // Matrix descriptor
*      hipsparseMatDescr_t descrA;
*      hipsparseCreateMatDescr(&descrA);
*
*      // Offload data to device
*      int* dAptr = NULL;
*      int* dAcol = NULL;
*      double*        dAval = NULL;
*      double*        dx    = NULL;
*      double*        dy    = NULL;
*
*      hipMalloc((void**)&dAptr, sizeof(int) * (m + 1));
*      hipMalloc((void**)&dAcol, sizeof(int) * nnz);
*      hipMalloc((void**)&dAval, sizeof(double) * nnz);
*      hipMalloc((void**)&dx, sizeof(double) * n);
*      hipMalloc((void**)&dy, sizeof(double) * m);
*
*      hipMemcpy(dAptr, hAptr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*      hipMemcpy(dAcol, hAcol, sizeof(int) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dAval, hAval, sizeof(double) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dx, hx, sizeof(double) * n, hipMemcpyHostToDevice);
*
*      // Convert CSR matrix to HYB format
*      hipsparseHybMat_t hybA;
*      hipsparseCreateHybMat(&hybA);
*
*      hipsparseDcsr2hyb(handle, m, n, descrA, dAval, dAptr, dAcol, hybA, 0, HIPSPARSE_HYB_PARTITION_AUTO);
*
*      // Clean up CSR structures
*      hipFree(dAptr);
*      hipFree(dAcol);
*      hipFree(dAval);
*
*      // Call hipsparse hybmv
*      hipsparseDhybmv(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &halpha, descrA, hybA, dx, &hbeta, dy);
*
*      // Copy result back to host
*      hipMemcpy(hy, dy, sizeof(double) * m, hipMemcpyDeviceToHost);
*
*      // Clear up on device
*      hipsparseDestroyHybMat(hybA);
*      hipsparseDestroyMatDescr(descrA);
*      hipsparseDestroy(handle);
*
*      hipFree(dx);
*      hipFree(dy);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseShybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDhybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseChybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZhybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_LEVEL2_HIPSPARSE_HYBMV_H */