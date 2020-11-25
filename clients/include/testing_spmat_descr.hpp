/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef TESTING_SPMAT_DESCR_HPP
#define TESTING_SPMAT_DESCR_HPP

#include "hipsparse_test_unique_ptr.hpp"

#include <hipsparse.h>

using namespace hipsparse_test;

void testing_spmat_descr_bad_arg(void)
{
    int64_t rows = 100;
    int64_t cols = 100;
    int64_t nnz  = 100;

    hipsparseIndexType_t rowType  = HIPSPARSE_INDEX_32I;
    hipsparseIndexType_t colType  = HIPSPARSE_INDEX_32I;
    hipsparseIndexBase_t idxBase  = HIPSPARSE_INDEX_BASE_ZERO;
    hipDataType          dataType = HIP_R_32F;
    hipsparseFormat_t    format   = HIPSPARSE_FORMAT_CSR;

    // Allocate memory on device
    auto row_data_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto col_data_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto val_data_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};

    int*   row_data = (int*)row_data_managed.get();
    int*   col_data = (int*)col_data_managed.get();
    float* val_data = (float*)val_data_managed.get();

    if(!row_data || !col_data || !val_data)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    hipsparseSpMatDescr_t A;

    // hipsparseCreateCoo
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCoo(
            nullptr, rows, cols, nnz, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCoo(
            &A, -1, cols, nnz, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Error: rows is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCoo(
            &A, rows, -1, nnz, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Error: cols is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCoo(
            &A, rows, cols, -1, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Error: nnz is < 0");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCoo(
            &A, rows, cols, nnz, nullptr, col_data, val_data, rowType, idxBase, dataType),
        "Error: row_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCoo(
            &A, rows, cols, nnz, row_data, nullptr, val_data, rowType, idxBase, dataType),
        "Error: col_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCoo(
            &A, rows, cols, nnz, row_data, col_data, nullptr, rowType, idxBase, dataType),
        "Error: val_data is nullptr");

    // hipsparseCreateCsr
    verify_hipsparse_status_invalid_pointer(hipsparseCreateCsr(nullptr,
                                                               rows,
                                                               cols,
                                                               nnz,
                                                               row_data,
                                                               col_data,
                                                               val_data,
                                                               rowType,
                                                               colType,
                                                               idxBase,
                                                               dataType),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCsr(
            &A, -1, cols, nnz, row_data, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: rows is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCsr(
            &A, rows, -1, nnz, row_data, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: cols is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCsr(
            &A, rows, cols, -1, row_data, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: nnz is < 0");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCsr(
            &A, rows, cols, nnz, nullptr, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: row_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCsr(
            &A, rows, cols, nnz, row_data, nullptr, val_data, rowType, colType, idxBase, dataType),
        "Error: col_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCsr(
            &A, rows, cols, nnz, row_data, col_data, nullptr, rowType, colType, idxBase, dataType),
        "Error: val_data is nullptr");

    // hipsparseDestroySpMat
    verify_hipsparse_status_invalid_pointer(hipsparseDestroySpMat(nullptr), "Error: A is nullptr");

    // Create valid descriptors
    hipsparseSpMatDescr_t coo;
    hipsparseSpMatDescr_t csr;

    verify_hipsparse_status_success(
        hipsparseCreateCoo(
            &coo, rows, cols, nnz, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Success");
    verify_hipsparse_status_success(hipsparseCreateCsr(&csr,
                                                       rows,
                                                       cols,
                                                       nnz,
                                                       row_data,
                                                       col_data,
                                                       val_data,
                                                       rowType,
                                                       colType,
                                                       idxBase,
                                                       dataType),
                                    "Success");

    void* row_ptr;
    void* col_ptr;
    void* val_ptr;

    // hipsparseCooGet
    verify_hipsparse_status_invalid_pointer(hipsparseCooGet(nullptr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, nullptr, &cols, &nnz, &row_ptr, &col_ptr, &val_ptr, &rowType, &idxBase, &dataType),
        "Error: rows is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, nullptr, &nnz, &row_ptr, &col_ptr, &val_ptr, &rowType, &idxBase, &dataType),
        "Error: cols is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCooGet(coo,
                                                            &rows,
                                                            &cols,
                                                            nullptr,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: nnz is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, &cols, &nnz, nullptr, &col_ptr, &val_ptr, &rowType, &idxBase, &dataType),
        "Error: row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, nullptr, &val_ptr, &rowType, &idxBase, &dataType),
        "Error: col_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, &col_ptr, nullptr, &rowType, &idxBase, &dataType),
        "Error: val_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, &col_ptr, &val_ptr, nullptr, &idxBase, &dataType),
        "Error: rowType is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, &col_ptr, &val_ptr, &rowType, nullptr, &dataType),
        "Error: idxBase is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, &col_ptr, &val_ptr, &rowType, &idxBase, nullptr),
        "Error: dataType is nullptr");

    // hipsparseCsrGet
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(nullptr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            nullptr,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: rows is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            nullptr,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: cols is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            nullptr,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: nnz is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            nullptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            nullptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: col_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            nullptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: val_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            nullptr,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: rowType is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            nullptr,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: colType is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            nullptr,
                                                            &dataType),
                                            "Error: idxBase is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            nullptr),
                                            "Error: dataType is nullptr");

    // hipsparseCsrSetPointers
    verify_hipsparse_status_invalid_pointer(
        hipsparseCsrSetPointers(nullptr, row_data, col_data, val_data), "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCsrSetPointers(csr, nullptr, col_data, val_data), "Error: row_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCsrSetPointers(csr, row_data, nullptr, val_data), "Error: col_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCsrSetPointers(csr, row_data, col_data, nullptr), "Error: val_data is nullptr");

    // hipsparseSpMatGetSize
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetSize(nullptr, &rows, &cols, &nnz),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetSize(coo, nullptr, &cols, &nnz),
                                            "Error: rows is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetSize(coo, &rows, nullptr, &nnz),
                                            "Error: cols is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetSize(coo, &rows, &cols, nullptr),
                                            "Error: nnz is nullptr");

    // hipsparseSpMatGetFormat
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetFormat(nullptr, &format),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetFormat(coo, nullptr),
                                            "Error: format is nullptr");

    // hipsparseSpMatGetIndexBase
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetIndexBase(nullptr, &idxBase),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetIndexBase(coo, nullptr),
                                            "Error: idxBase is nullptr");

    // hipsparseSpMatGetValues
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetValues(nullptr, &val_ptr),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetValues(coo, nullptr),
                                            "Error: val_ptr is nullptr");

    // hipsparseSpMatSetValues
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatSetValues(nullptr, val_ptr),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatSetValues(coo, nullptr),
                                            "Error: val_ptr is nullptr");

    // Destroy valid descriptors
    verify_hipsparse_status_success(hipsparseDestroySpMat(coo), "Success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(csr), "Success");
}

#endif // TESTING_SPMAT_DESCR_HPP
