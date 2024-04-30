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
#ifndef HIPSPARSE_CONVERSION_HIPSPARSE_COO2CSR_H
#define HIPSPARSE_CONVERSION_HIPSPARSE_COO2CSR_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
 *  \brief Convert a sparse COO matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXcoo2csr converts the COO array containing the row indices into a
 *  CSR array of row offsets, that point to the start of every row.
 *  It is assumed that the COO row index array is sorted.
 *
 *  \note It can also be used, to convert a COO array containing the column indices into
 *  a CSC array of column offsets, that point to the start of every column. Then, it is
 *  assumed that the COO column index array is sorted, instead.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t    handle,
                                    const int*           cooRowInd,
                                    int                  nnz,
                                    int                  m,
                                    int*                 csrRowPtr,
                                    hipsparseIndexBase_t idxBase);

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CONVERSION_HIPSPARSE_COO2CSR_H */