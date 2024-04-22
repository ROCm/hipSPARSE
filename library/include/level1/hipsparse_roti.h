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
#ifndef HIPSPARSE_LEVEL1_HIPSPARSE_ROTI_H
#define HIPSPARSE_LEVEL1_HIPSPARSE_ROTI_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level1_module
*  \brief Apply Givens rotation to a dense and a sparse vector.
*
*  \details
*  \p hipsparseXroti applies the Givens rotation matrix \f$G\f$ to the sparse vector
*  \f$x\f$ and the dense vector \f$y\f$, where
*  \f[
*    G = \begin{pmatrix} c & s \\ -s & c \end{pmatrix}
*  \f]
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          x_tmp = x_val[i];
*          y_tmp = y[x_ind[i]];
*
*          x_val[i]    = c * x_tmp + s * y_tmp;
*          y[x_ind[i]] = c * y_tmp - s * x_tmp;
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \par Example
*  \code{.c}
*      // Number of non-zeros of the sparse vector
*      int nnz = 3;
*
*      // Sparse index vector
*      int hx_ind[3] = {0, 3, 5};
*
*      // Sparse value vector
*      float hx_val[3] = {1.0f, 2.0f, 3.0f};
*
*      // Dense vector
*      float hy[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
*
*      // c and s
*      float c = 3.7;
*      float s = 1.3;
*
*      // Index base
*      hipsparseIndexBase_t idx_base = HIPSPARSE_INDEX_BASE_ZERO;
*
*      // Offload data to device
*      int* dx_ind;
*      float*        dx_val;
*      float*        dy;
*
*      hipMalloc((void**)&dx_ind, sizeof(int) * nnz);
*      hipMalloc((void**)&dx_val, sizeof(float) * nnz);
*      hipMalloc((void**)&dy, sizeof(float) * 9);
*
*      hipMemcpy(dx_ind, hx_ind, sizeof(int) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dx_val, hx_val, sizeof(float) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dy, hy, sizeof(float) * 9, hipMemcpyHostToDevice);
*
*      // hipSPARSE handle
*      hipsparseHandle_t handle;
*      hipsparseCreate(&handle);
*
*      // Call sroti
*      hipsparseSroti(handle, nnz, dx_val, dx_ind, dy, &c, &s, idx_base);
*
*      // Copy result back to host
*      hipMemcpy(hx_val, dx_val, sizeof(float) * nnz, hipMemcpyDeviceToHost);
*      hipMemcpy(hy, dy, sizeof(float) * 9, hipMemcpyDeviceToHost);
*
*      // Clear hipSPARSE
*      hipsparseDestroy(handle);
*
*      // Clear device memory
*      hipFree(dx_ind);
*      hipFree(dx_val);
*      hipFree(dy);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSroti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 float*               xVal,
                                 const int*           xInd,
                                 float*               y,
                                 const float*         c,
                                 const float*         s,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDroti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 double*              xVal,
                                 const int*           xInd,
                                 double*              y,
                                 const double*        c,
                                 const double*        s,
                                 hipsparseIndexBase_t idxBase);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_LEVEL1_HIPSPARSE_ROTI_H */