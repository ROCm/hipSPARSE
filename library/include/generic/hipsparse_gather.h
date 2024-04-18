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

/*! \ingroup generic_module
*  \brief Description: Gather elements from a dense vector and store them into a sparse vector.
*
*  \details
*  \ref hipsparseGather gathers the elements from the dense vector \f$y\f$ and stores
*  them in the sparse vector \f$x\f$.
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          x_val[i] = y[x_ind[i]];
*      }
*  \endcode
*
*  \par Example
*  \code{.c}
*    // Number of non-zeros of the sparse vector
*    int nnz = 3;
*
*    // Size of sparse and dense vector
*    int size = 9;
*
*    // Sparse index vector
*    std::vector<int> hx_ind = {0, 3, 5};
*
*    // Dense vector
*    std::vector<float> hy = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
*
*    // Offload data to device
*    int* dx_ind;
*    float* dx_val;
*    float* dy;
*    hipMalloc((void**)&dx_ind, sizeof(int) * nnz);
*    hipMalloc((void**)&dx_val, sizeof(float) * nnz);
*    hipMalloc((void**)&dy, sizeof(float) * size);
*
*    hipMemcpy(dx_ind, hx_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dy, hy.data(), sizeof(float) * size, hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Create sparse vector X
*    hipsparseSpVecDescr_t vecX;
*    hipsparseCreateSpVec(&vecX,
*                         size,
*                         nnz,
*                         dx_ind,
*                         dx_val,
*                         HIPSPARSE_INDEX_32I,
*                         HIPSPARSE_INDEX_BASE_ZERO,
*                         HIP_R_32F);
*
*    // Create dense vector Y
*    hipsparseDnVecDescr_t vecY;
*    hipsparseCreateDnVec(&vecY, size, dy, HIP_R_32F);
*
*    // Perform gather
*    hipsparseGather(handle, vecY, vecX);
*
*    hipsparseSpVecGetValues(vecX, (void**)&dx_val);
*
*    // Copy result back to host
*    std::vector<float> hx_val(nnz, 0.0f);
*    hipMemcpy(hx_val.data(), dx_val, sizeof(float) * nnz, hipMemcpyDeviceToHost);
*
*    // Clear hipSPARSE
*    hipsparseDestroySpVec(vecX);
*    hipsparseDestroyDnVec(vecY);
*    hipsparseDestroy(handle);
*
*    // Clear device memory
*    hipFree(dx_ind);
*    hipFree(dx_val);
*    hipFree(dy);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGather(hipsparseHandle_t          handle,
                                  hipsparseConstDnVecDescr_t vecY,
                                  hipsparseSpVecDescr_t      vecX);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGather(hipsparseHandle_t     handle,
                                  hipsparseDnVecDescr_t vecY,
                                  hipsparseSpVecDescr_t vecX);
#endif

#ifdef __cplusplus
}
#endif