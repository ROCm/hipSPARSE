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
#ifndef HIPSPARSE_GENERIC_HIPSPARSE_SPVV_H
#define HIPSPARSE_GENERIC_HIPSPARSE_SPVV_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Description: Compute the inner dot product of a sparse vector with a dense vector
*
*  \details
*  \p hipsparseSpVV_bufferSize computes the required user allocated buffer size needed when computing the 
*  inner dot product of a sparse vector with a dense vector
*
*  See full example below
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t          handle,
                                           hipsparseOperation_t       opX,
                                           hipsparseConstSpVecDescr_t vecX,
                                           hipsparseConstDnVecDescr_t vecY,
                                           void*                      result,
                                           hipDataType                computeType,
                                           size_t*                    bufferSize);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t     handle,
                                           hipsparseOperation_t  opX,
                                           hipsparseSpVecDescr_t vecX,
                                           hipsparseDnVecDescr_t vecY,
                                           void*                 result,
                                           hipDataType           computeType,
                                           size_t*               bufferSize);
#endif

/*! \ingroup generic_module
*  \brief Description: Compute the inner dot product of a sparse vector with a dense vector
*
*  \details
*  \p hipsparseSpVV computes the inner dot product of a sparse vector with a dense vector. This routine takes a user 
*  allocated buffer whose size must first be computed by calling \p hipsparseSpVV_bufferSize
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
*    // Sparse value vector
*    std::vector<float> hx_val = {1.0f, 2.0f, 3.0f};
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
*    hipMemcpy(dx_val, hx_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dy, hy.data(), sizeof(float) * size, hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Create sparse vector X
*    hipsparseSpVecDescr_t vecX;
*    hipsparseCreateSpVec(&vecX,
*                        size,
*                        nnz,
*                        dx_ind,
*                        dx_val,
*                        HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO,
*                        HIP_R_32F);
*
*    // Create dense vector Y
*    hipsparseDnVecDescr_t vecY;
*    hipsparseCreateDnVec(&vecY, size, dy, HIP_R_32F);
*
*    // Obtain buffer size
*    float hresult = 0.0f;
*    size_t buffer_size;
*    hipsparseSpVV_bufferSize(handle,
*                HIPSPARSE_OPERATION_NON_TRANSPOSE,
*                vecX,
*                vecY,
*                &hresult,
*                HIP_R_32F,
*                &buffer_size);
*
*    void* temp_buffer;
*    hipMalloc(&temp_buffer, buffer_size);
*
*    // SpVV
*    hipsparseSpVV(handle,
*                HIPSPARSE_OPERATION_NON_TRANSPOSE,
*                vecX,
*                vecY,
*                &hresult,
*                HIP_R_32F,
*                temp_buffer);
*
*    hipDeviceSynchronize();
*
*    std::cout << "hresult: " << hresult << std::endl;
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
*    hipFree(temp_buffer);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t          handle,
                                hipsparseOperation_t       opX,
                                hipsparseConstSpVecDescr_t vecX,
                                hipsparseConstDnVecDescr_t vecY,
                                void*                      result,
                                hipDataType                computeType,
                                void*                      externalBuffer);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t     handle,
                                hipsparseOperation_t  opX,
                                hipsparseSpVecDescr_t vecX,
                                hipsparseDnVecDescr_t vecY,
                                void*                 result,
                                hipDataType           computeType,
                                void*                 externalBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_GENERIC_HIPSPARSE_SPVV_H */