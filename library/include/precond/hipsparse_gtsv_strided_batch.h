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

/*! \ingroup precond_module
*  \brief Strided Batch tridiagonal solver (no pivoting)
*
*  \details
*  \p hipsparseXgtsv2StridedBatch_bufferSizeExt returns the size of the temporary storage
*  buffer in bytes that is required by hipsparseXgtsv2StridedBatch(). The temporary 
*  storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const float*      dl,
                                                            const float*      d,
                                                            const float*      du,
                                                            const float*      x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const double*     dl,
                                                            const double*     d,
                                                            const double*     du,
                                                            const double*     x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const hipComplex* dl,
                                                            const hipComplex* d,
                                                            const hipComplex* du,
                                                            const hipComplex* x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                            int                     m,
                                                            const hipDoubleComplex* dl,
                                                            const hipDoubleComplex* d,
                                                            const hipDoubleComplex* du,
                                                            const hipDoubleComplex* x,
                                                            int                     batchCount,
                                                            int                     batchStride,
                                                            size_t* pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Strided Batch tridiagonal solver (no pivoting)
*
*  \details
*  \p hipsparseXgtsv2StridedBatch solves a batched tridiagonal linear system
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const float*      dl,
                                              const float*      d,
                                              const float*      du,
                                              float*            x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const double*     dl,
                                              const double*     d,
                                              const double*     du,
                                              double*           x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const hipComplex* dl,
                                              const hipComplex* d,
                                              const hipComplex* du,
                                              hipComplex*       x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2StridedBatch(hipsparseHandle_t       handle,
                                              int                     m,
                                              const hipDoubleComplex* dl,
                                              const hipDoubleComplex* d,
                                              const hipDoubleComplex* du,
                                              hipDoubleComplex*       x,
                                              int                     batchCount,
                                              int                     batchStride,
                                              void*                   pBuffer);
/**@}*/

#ifdef __cplusplus
}
#endif
