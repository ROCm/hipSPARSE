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
*  \brief Convert a sparse GEneral BSR matrix into a sparse GEneral BSC matrix
*
*  \details
*  \p hipsparseXgebsr2gebsc_bufferSize returns the size of the temporary storage buffer
*  required by hipsparseXgebsr2gebsc().
*  The temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const float*      bsr_val,
                                                   const int*        bsr_row_ptr,
                                                   const int*        bsr_col_ind,
                                                   int               row_block_dim,
                                                   int               col_block_dim,
                                                   size_t*           p_buffer_size);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const double*     bsr_val,
                                                   const int*        bsr_row_ptr,
                                                   const int*        bsr_col_ind,
                                                   int               row_block_dim,
                                                   int               col_block_dim,
                                                   size_t*           p_buffer_size);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const hipComplex* bsr_val,
                                                   const int*        bsr_row_ptr,
                                                   const int*        bsr_col_ind,
                                                   int               row_block_dim,
                                                   int               col_block_dim,
                                                   size_t*           p_buffer_size);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsc_bufferSize(hipsparseHandle_t       handle,
                                                   int                     mb,
                                                   int                     nb,
                                                   int                     nnzb,
                                                   const hipDoubleComplex* bsr_val,
                                                   const int*              bsr_row_ptr,
                                                   const int*              bsr_col_ind,
                                                   int                     row_block_dim,
                                                   int                     col_block_dim,
                                                   size_t*                 p_buffer_size);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse GEneral BSR matrix into a sparse GEneral BSC matrix
*
*  \details
*  \p hipsparseXgebsr2gebsc converts a GEneral BSR matrix into a GEneral BSC matrix. \p hipsparseXgebsr2gebsc
*  can also be used to convert a GEneral BSC matrix into a GEneral BSR matrix. \p copy_values decides
*  whether \p bsc_val is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
*  or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
*
*  \p hipsparseXgebsr2gebsc requires extra temporary storage buffer that has to be allocated
*  by the user. Storage buffer size can be determined by hipsparseXgebsr2gebsc_bufferSize().
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const float*         bsr_val,
                                        const int*           bsr_row_ptr,
                                        const int*           bsr_col_ind,
                                        int                  row_block_dim,
                                        int                  col_block_dim,
                                        float*               bsc_val,
                                        int*                 bsc_row_ind,
                                        int*                 bsc_col_ptr,
                                        hipsparseAction_t    copy_values,
                                        hipsparseIndexBase_t idx_base,
                                        void*                temp_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const double*        bsr_val,
                                        const int*           bsr_row_ptr,
                                        const int*           bsr_col_ind,
                                        int                  row_block_dim,
                                        int                  col_block_dim,
                                        double*              bsc_val,
                                        int*                 bsc_row_ind,
                                        int*                 bsc_col_ptr,
                                        hipsparseAction_t    copy_values,
                                        hipsparseIndexBase_t idx_base,
                                        void*                temp_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const hipComplex*    bsr_val,
                                        const int*           bsr_row_ptr,
                                        const int*           bsr_col_ind,
                                        int                  row_block_dim,
                                        int                  col_block_dim,
                                        hipComplex*          bsc_val,
                                        int*                 bsc_row_ind,
                                        int*                 bsc_col_ptr,
                                        hipsparseAction_t    copy_values,
                                        hipsparseIndexBase_t idx_base,
                                        void*                temp_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsc(hipsparseHandle_t       handle,
                                        int                     mb,
                                        int                     nb,
                                        int                     nnzb,
                                        const hipDoubleComplex* bsr_val,
                                        const int*              bsr_row_ptr,
                                        const int*              bsr_col_ind,
                                        int                     row_block_dim,
                                        int                     col_block_dim,
                                        hipDoubleComplex*       bsc_val,
                                        int*                    bsc_row_ind,
                                        int*                    bsc_col_ptr,
                                        hipsparseAction_t       copy_values,
                                        hipsparseIndexBase_t    idx_base,
                                        void*                   temp_buffer);
/**@}*/

#ifdef __cplusplus
}
#endif