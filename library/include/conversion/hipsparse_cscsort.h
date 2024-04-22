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

/*! \ingroup conv_module
*  \brief Sort a sparse CSC matrix
*
*  \details
*  \p hipsparseXcscsort_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes required by hipsparseXcscsort(). The temporary storage buffer must be 
*  allocated by the user.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcscsort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        cscColPtr,
                                                  const int*        cscRowInd,
                                                  size_t*           pBufferSizeInBytes);

/*! \ingroup conv_module
*  \brief Sort a sparse CSC matrix
*
*  \details
*  \p hipsparseXcscsort sorts a matrix in CSC format. The sorted permutation vector
*  \p perm can be used to obtain sorted \p csc_val array. In this case, \p perm must be
*  initialized as the identity permutation, see hipsparseCreateIdentityPermutation().
*
*  \p hipsparseXcscsort requires extra temporary storage buffer that has to be allocated by
*  the user. Storage buffer size can be determined by hipsparseXcscsort_bufferSizeExt().
*
*  \note
*  \p perm can be \p NULL if a sorted permutation vector is not required.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcscsort(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    const int*                cscColPtr,
                                    int*                      cscRowInd,
                                    int*                      P,
                                    void*                     pBuffer);

#ifdef __cplusplus
}
#endif
